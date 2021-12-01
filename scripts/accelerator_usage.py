import argparse
from accelerate import Accelerator
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    args = parser.parse_args()
    # Sanity checks
    if args.per_device_train_batch_size is None:
        raise ValueError("Need a per_device_train_batch_size.")

    return args

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    os.environ['NVIDIA_VISIBLE_DEVICES'] = args.cuda_devices
    # Initialize the accelerator. Let the accelerator handle device placement.
    accelerator = Accelerator()
    args.device = accelerator.device
    """  
    tokenizer, model = get_model(args, num_labels)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=eval_data_collator, batch_size=args.per_device_eval_batch_size)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)    

    """

class Trainer():
    def __init__(self,
                args,
                model,
                optimizer,
                lr_scheduler,
                train_dataloader,
                eval_dataloader,
                external_dataloader,
                logger,
                accelerator,
                metric,
                label_list,
                tokenizer,
                from_checkpoint=None,
                test_dataloader=None,
                ) -> None:
        self.accelerator = accelerator

    def compute_loss(self, ratio):
        self.model.train()
        batch = self._get_batch(ratio)
        outputs = self.model(**batch)
        loss = outputs.loss
        loss = loss / self.args.gradient_accumulation_steps
        self.accelerator.backward(loss)
        return loss.item()

    def update(self, tr_loss, loss_step):
        if self.completed_steps % self.args.steps_to_log == 0:
            self.logger.info(
                "step {}, learning rate {}, average loss {}".format(
                    self.completed_steps,
                    self.optimizer.param_groups[0]["lr"],
                    tr_loss / loss_step
                )
            )
            if self.accelerator.is_main_process:
                if self.writter is not None:
                    self.writter.add_scalar('train/loss', tr_loss / loss_step, self.completed_steps)
                    self.writter.add_scalar('train/lr', self.optimizer.param_groups[0]["lr"], self.completed_steps)
            tr_loss = 0.0
            loss_step = 0
        
        if self.completed_steps % self.args.steps_to_eval == 0:
            self.evaluate()
        
        if self.completed_steps % self.args.steps_to_save == 0:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self._save_model(
                    save_path = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(self.completed_steps))
                )
                # delete outdated checkpoints
                for files in os.listdir(self.args.output_dir):
                    file_name = os.path.join(self.args.output_dir, files)
                    if os.path.isdir(file_name) and files.startswith('checkpoint-'):
                        checked_step = int(files[11:])
                        if self.completed_steps - checked_step >= self.args.max_ckpts_to_keep * self.args.steps_to_save:
                            if self.accelerator.is_main_process:
                                shutil.rmtree(file_name)

if __name__ == "__main__":
    main()

"""  
1) import and initialize
from accelerate import Accelerator
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    os.environ['NVIDIA_VISIBLE_DEVICES'] = args.cuda_devices
    # Initialize the accelerator. Let the accelerator handle device placement.
    accelerator = Accelerator()
    args.device = accelerator.device

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

2) under accelerator.is_main_process to handle log
    if accelerator.is_main_process:
        if os.path.exists(logfile):
            os.remove(logfile)
        os.mknod(logfile)
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.setLevel(logging.INFO if accelerator.is_main_process else logging.ERROR)
    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if self.accelerator.is_main_process:
        if self.writter is not None:
            self.writter.add_scalar('train/loss', tr_loss / loss_step, self.completed_steps)
            self.writter.add_scalar('train/lr', self.optimizer.param_groups[0]["lr"], self.completed_steps)

3) accelerator to backward loss
    def compute_loss(self, ratio):
        self.model.train()
        batch = self._get_batch(ratio)
        outputs = self.model(**batch)
        loss = outputs.loss
        loss = loss / self.args.gradient_accumulation_steps
        self.accelerator.backward(loss)
        return loss.item()

4) under accelerator.wait_for_everyone() to save model and remove checkpoints
    self.accelerator.wait_for_everyone()
    if self.accelerator.is_main_process:
        self._save_model(
            save_path = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(self.completed_steps))
        )
        # delete outdated checkpoints
        for files in os.listdir(self.args.output_dir):
            file_name = os.path.join(self.args.output_dir, files)
            if os.path.isdir(file_name) and files.startswith('checkpoint-'):
                checked_step = int(files[11:])
                if self.completed_steps - checked_step >= self.args.max_ckpts_to_keep * self.args.steps_to_save:
                    if self.accelerator.is_main_process:
                        shutil.rmtree(file_name)

"""