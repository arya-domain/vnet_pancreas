from validate import *
from Utils.losses import *
import torch.nn.functional as F
from Model.Vnet import VNet
from Utils.lr_scheduler import PolyLR


def create_model(ema=False):
    # Network definition
    net = VNet(n_channels=1, n_classes=2, normalization="batchnorm", has_dropout=True)

    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def softmax_mse_loss(inputs, targets):

    # assert input.size()==targets.size()
    output = (inputs - targets)**2

    return output


class NegEntropy(object):
    def __call__(self, outputs_list):
        entropy_results = torch.tensor([0.0]).cuda()
        for i in outputs_list:
            current_prob = torch.softmax(i, dim=0).clamp(min=1e-6, max=1.0)
            entropy_results += torch.sum(
                current_prob.log() * current_prob, dim=0
            ).mean()
        return entropy_results / len(outputs_list)


class Trainer:
    def __init__(self, config_, train_loader, valid_set, logger=None, my_wandb=None):
        self.config = config_
        # labelled_num == unlabelled_num within a batch
        self.labeled_num = int(self.config.batch_size / 2)
        super(Trainer, self).__init__()
        if not self.config.ddp_training:
            self.model1 = create_model()
            self.model2 = create_model()
        else:
            raise NotImplementedError

        self.optimizer1 = torch.optim.SGD(
            self.model1.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

        self.optimizer2 = torch.optim.SGD(
            self.model2.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

        self.lr_scheduler = PolyLR(
            start_lr=self.config.learning_rate,
            lr_power=0.9,
            total_iters=self.config.max_iterations,
        )

        self.iter_num_count = 0
        self.train_loader = train_loader
        self.val_set = valid_set
        self.logger = logger
        self.my_wandb = my_wandb

        # specify loss function in use
        self.sup_ce_loss = F.cross_entropy
        self.sup_dice_loss = dice_loss
        self.weight_scheduler = ConsistencyWeight(config=self.config)
        self.saddle_reg = NegEntropy()  # NegProb()

    @staticmethod
    def rand_bbox(size, lam=None):
        # past implementation
        W = size[2]
        H = size[3]
        B = size[0]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        cx = np.random.randint(
            size=[
                B,
            ],
            low=int(W / 8),
            high=W,
        )
        cy = np.random.randint(
            size=[
                B,
            ],
            low=int(H / 8),
            high=H,
        )
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    def cut_mix(self, volume=None, mask=None, gt=None):
        mix_volume = volume.clone()
        mix_target = mask.clone()
        mix_gt = gt.clone()
        u_rand_index = torch.randperm(volume.size()[0])[: volume.size()[0]].cuda()
        u_bbx1, u_bby1, u_bbx2, u_bby2 = self.rand_bbox(
            volume.size(), lam=np.random.beta(4, 4)
        )
        for i in range(0, self.config.batch_size * 2):
            if i < mix_volume.shape[0]:
                mix_volume[i, :, u_bbx1[i] : u_bbx2[i], u_bby1[i] : u_bby2[i]] = volume[
                    u_rand_index[i], :, u_bbx1[i] : u_bbx2[i], u_bby1[i] : u_bby2[i]
                ]
                mix_target[i, :, u_bbx1[i] : u_bbx2[i], u_bby1[i] : u_bby2[i]] = mask[
                    u_rand_index[i], :, u_bbx1[i] : u_bbx2[i], u_bby1[i] : u_bby2[i]
                ]
                mix_gt[i, u_bbx1[i] : u_bbx2[i], u_bby1[i] : u_bby2[i]] = mix_gt[
                    u_rand_index[i], u_bbx1[i] : u_bbx2[i], u_bby1[i] : u_bby2[i]
                ]

        return mix_volume, mix_target, mix_gt

    def calculate_current_acc(self, preds, label_batch):
        mask = torch.softmax(preds, dim=1).max(1)[0] > self.config.threshold
        y_tilde = torch.argmax(preds, dim=1)
        current_acc = (y_tilde[mask] == label_batch[mask]).sum() / len(
            label_batch[mask].ravel()
        )
        return current_acc

    def train_epoch(self, model1, model2, optimizer1, optimizer2, iter_num=0):

        for batch_idx, (normal_batch, cons_batch) in enumerate(self.train_loader):
            if iter_num >= self.config.max_iterations:
                return iter_num
            else:
                iter_num = iter_num + 1
            volume_batch, label_batch, tokens_batch = (
                normal_batch["data"],
                normal_batch["label"],
                normal_batch["tokens"],
            )
            volume_batch, label_batch, tokens_batch = (
                volume_batch.cuda(),
                label_batch.cuda(),
                tokens_batch.cuda(),
            )
            gt_ratio = torch.sum(label_batch == 0) / len(torch.ravel(label_batch))

            model1.train()
            model2.train()
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            consistency_weight = self.weight_scheduler(iter_num)
            """ model1 training """
            # spatial-consistent
            noise = torch.zeros_like(volume_batch).uniform_(-0.05, 0.05)
            sup_outputs1, _ = model1(
                (volume_batch + noise).clamp(min=0.0, max=1.0), tokens_batch
            )

            """ supervised part """
            # calculate the loss
            sup_ce_loss1 = F.cross_entropy(
                sup_outputs1[: self.labeled_num], label_batch[: self.labeled_num]
            ).mean()
            outputs_soft = F.softmax(sup_outputs1, dim=1)
            sup_dice_loss1 = self.sup_dice_loss(
                outputs_soft[: self.labeled_num, 1, :, :, :],
                label_batch[: self.labeled_num] == 1,
            ).mean()
            loss1_sup = (sup_ce_loss1 + sup_dice_loss1) * 0.5 

            """ unsupervised part """
            # pseudo-label
            with torch.no_grad():
                pseudo_label1, _ = model2(volume_batch, tokens_batch)

            noise = torch.zeros_like(volume_batch).uniform_(-0.05, 0.05)
            unsup_outputs1, _ = model1(
                (volume_batch + noise).clamp(min=0.0, max=1.0), tokens_batch
            )
            loss1_crc, _ = semi_crc_loss(
                inputs=unsup_outputs1[self.labeled_num :],
                targets=pseudo_label1[self.labeled_num :],
                threshold=self.config.threshold,
                neg_threshold=0.1,
                conf_mask=True,
            )

            # the overall semi-supervised loss
            loss1_unsup = loss1_crc
            loss1 = loss1_sup + loss1_unsup * consistency_weight

            """ model2 training """
            # spatial-consistent
            noise = torch.zeros_like(volume_batch).uniform_(-0.05, 0.05)
            sup_outputs2, _ = model2(
                (volume_batch + noise).clamp(min=0.0, max=1.0), tokens_batch
            )

            """ supervised part """
            # calculate the loss
            sup_ce_loss2 = F.cross_entropy(
                sup_outputs2[: self.labeled_num], label_batch[: self.labeled_num]
            ).mean()
            outputs_soft = F.softmax(sup_outputs2, dim=1)
            sup_dice_loss2 = self.sup_dice_loss(
                outputs_soft[: self.labeled_num, 1, :, :, :],
                label_batch[: self.labeled_num] == 1,
            ).mean()
            loss2_sup =  (sup_ce_loss2 + sup_dice_loss2) * 0.5 

            # ####################################### Dice Loss #######################################
            # v_outputs_soft2 = F.softmax(sup_outputs1, dim=1)
            # r_outputs_soft2 = F.softmax(sup_outputs2, dim=1)
            # # Dice loss
            # v_predict = torch.max(
            #     v_outputs_soft2[: self.labeled_num, :, :, :, :],
            #     1,
            # )[1]
            # r_predict = torch.max(
            #     r_outputs_soft2[: self.labeled_num, :, :, :, :],
            #     1,
            # )[1]
            # diff_mask = ((v_predict == 1) ^ (r_predict == 1)).to(torch.int32)
            # v_mse_dist = softmax_mse_loss(
            #     v_outputs_soft2[: self.labeled_num, 1, :, :, :],
            #     label_batch[: self.labeled_num],
            # ).mean()
            # r_mse_dist = softmax_mse_loss(
            #     r_outputs_soft2[: self.labeled_num, 1, :, :, :],
            #     label_batch[: self.labeled_num],
            # ).mean()
            # v_mse = torch.sum(diff_mask * v_mse_dist) / (torch.sum(diff_mask) + 1e-16)
            # r_mse = torch.sum(diff_mask * r_mse_dist) / (torch.sum(diff_mask) + 1e-16)

            # loss1_sup =  (sup_ce_loss1 + sup_dice_loss1) + 0.2 * v_mse * 0.5 
            # loss2_sup =  (sup_ce_loss2 + sup_dice_loss2) + 0.2 * r_mse * 0.5 
            # #######################################  ################################################

            """ unsupervised part """
            # pseudo-label
            with torch.no_grad():
                pseudo_label2, _ = model1(volume_batch, tokens_batch)
            cut_mix_input2, cut_mix_output2, label_batch = self.cut_mix(
                volume_batch, pseudo_label2, gt=label_batch
            )

            noise = torch.zeros_like(volume_batch).uniform_(-0.05, 0.05)
            unsup_outputs2, _ = model2(
                (cut_mix_input2 + noise).clamp(min=0.0, max=1.0), tokens_batch
            )

            loss2_crc, _ = semi_crc_loss(
                inputs=unsup_outputs2[self.labeled_num :],
                targets=cut_mix_output2[self.labeled_num :],
                threshold=self.config.threshold,
                neg_threshold=0.1,
                conf_mask=True,
            )

            # the over all semi-supervised loss
            loss2_unsup = loss2_crc
            loss2 = loss2_sup + loss2_unsup * consistency_weight

            # update 2 networks' learning rate
            current_lr = self.lr_scheduler.get_lr(cur_iter=iter_num)
            for _, opt_group in enumerate(optimizer1.param_groups):
                opt_group["lr"] = current_lr
            for _, opt_group in enumerate(optimizer2.param_groups):
                opt_group["lr"] = current_lr

            # update 2 networks' gradients
            loss1.backward()
            loss2.backward()

            optimizer1.step()
            optimizer2.step()

            if batch_idx % 8 == 0:
                pred_ratio = torch.mean(
                    torch.softmax(pseudo_label1, dim=1)[:, 0]
                ).item()
                pseudo_label1 = torch.argmax(pseudo_label1, dim=1)
                current_acc_t = self.calculate_current_acc(
                    preds=cut_mix_output2, label_batch=label_batch
                )
                info_dict = {
                    "model1_ce_loss": sup_ce_loss1.mean().item(),
                    "model1_dice_loss": sup_dice_loss1.mean().item(),
                    "model2_ce_loss": sup_ce_loss2.mean().item(),
                    "model2_dice_loss": sup_dice_loss2.mean().item(),
                    "model1_pseudo-acc. > {}".format(
                        str(self.config.threshold)
                    ): current_acc_t,
                    "model1_prob_mean": pred_ratio,
                    "model1_hard_label_ratio": torch.sum(pseudo_label1 == 0)
                    / len(torch.ravel(pseudo_label1)),
                    "model1_gt_label_ratio": gt_ratio.item(),
                    "cons. weight": consistency_weight,
                }

                self.my_wandb.upload_wandb_info(info_dict=info_dict)

            if iter_num % 100 == 0:
                self.my_wandb.produce_2d_slice(
                    image=cut_mix_input2, pred=cut_mix_output2, label=label_batch
                )

            self.logger.info(
                "model1 iteration %d : loss : %f unsup_loss: %f, loss_ce: %f loss_dice: %f"
                % (
                    iter_num,
                    loss1.item(),
                    loss1_crc.item(),
                    sup_ce_loss1.item(),
                    sup_dice_loss1.item(),
                )
            )

            self.logger.info(
                "model2 iteration %d : loss : %f unsup_loss: %f, loss_ce: %f loss_dice: %f"
                % (
                    iter_num,
                    loss2.item(),
                    loss2_crc.item(),
                    sup_ce_loss2.item(),
                    sup_dice_loss2.item(),
                )
            )
        self.logger.info("------------------------------------------------------- \n")
        return iter_num

    def validate(self, epoch, model, model_id):
        metric_record = 0.0
        self.logger.critical("{} eval time ...".format(model_id))
        model.eval()
        assert self.val_set.aug is False, ">> no augmentation for eval. set"
        dataloader = iter(self.val_set)
        tbar = range(len(self.val_set))
        tbar = tqdm(tbar, ncols=135)
        for batch_idx in tbar:
            x, y, t = next(dataloader)
            y_tilde, y_hat = test_single_case(
                net=model,
                image=x,
                tokens=t,
                stride_xy=16,
                stride_z=16,
                patch_size=(96, 96, 96),
                num_classes=self.config.num_classes,
            )
            if np.sum(y_tilde) == 0:
                single_metric = (0, 0, 0, 0)
            else:
                single_metric = calculate_metric_percase(
                    numpy.array(y_tilde), numpy.array(y[:])
                )

            metric_record += np.asarray(single_metric)
            if batch_idx == 0 and epoch >= self.config.n_epochs - 1:
                self.my_wandb.produce_3d_gif(result=y, name="y", current_epoch=epoch)
                self.my_wandb.produce_3d_gif(
                    result=y_tilde, name="y_tilde", current_epoch=epoch, color="blue"
                )

        metric_record = metric_record / len(self.val_set)
        info = {
            "{}_dice".format(model_id): metric_record[0],
            "{}_jaccard".format(model_id): metric_record[1],
            "{}_hd95".format(model_id): metric_record[2],
            "{}_asd".format(model_id): metric_record[3],
        }

        self.my_wandb.upload_wandb_info(info_dict=info)
        if self.config.save_ckpt:
            self.my_wandb.save_ckpt(
                model,
                model_id
                + str(info["{}_dice".format(model_id)])
                + "_"
                + str(epoch)
                + ".pth",
            )

        self.logger.critical(
            "valid | dice: {}, "
            "jaccard: {}, "
            "hd95: {}"
            "asd: {}".format(
                metric_record[0], metric_record[1], metric_record[2], metric_record[3]
            )
        )

        return

    def run(self):
        for epoch in range(0, self.config.n_epochs):
            self.iter_num_count = self.train_epoch(
                model1=self.model1,
                optimizer1=self.optimizer1,
                model2=self.model2,
                optimizer2=self.optimizer2,
                iter_num=self.iter_num_count,
            )
            # 1). last several iterations evaluation protocol
            # 2). best evaluation protocol
            # if self.iter_num_count >= self.config.max_iterations or epoch >= (self.config.n_epochs - self.config.last_val_epochs):
            # if self.iter_num_count >= 3: # brute
            if self.iter_num_count >= 800 and epoch % 25 == 0:
                self.validate(epoch, self.model1, model_id="model1")
                self.validate(epoch, self.model2, model_id="model2")
        return
