from typing import Iterable, Dict, Any
import logging
import os
import argparse
import math
from sys import path
from allennlp.common import Params
from allennlp.common.util import prepare_environment, prepare_global_logging, cleanup_global_logging
from allennlp.nn import util as nn_util
from allennlp.data import DatasetReader, Instance
from allennlp.data import Vocabulary, DataIterator
from allennlp.training import Trainer
from allennlp.models import Model
# from allennlp.training.util import evaluate
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common.tqdm import Tqdm
from models import GenderClassifier, ProfessionClassifier, MainAdversarial
from dataset_reader import FastTextProfessionGenderDataReader

def main(args):
    params = Params.from_file(args.config_path)
    print(args.output_dir)
    stdout_handler = prepare_global_logging(args.output_dir,False)
    prepare_environment(params)

    reader = DatasetReader.from_params(params["dataset_reader"])
    train_dataset = reader.read(params.pop("train_protected"))
    valid_dataset = reader.read(params.pop("validate_protected"))
    test_dataset = reader.read(params.pop("test_protected"))
    vocab = Vocabulary.from_instances(train_dataset + valid_dataset + test_dataset)
    vocab.save_to_files(output_path+'/vocab')
    #set model from config file
    model_params = params.pop("model", None)
    model = Model.from_params(model_params.duplicate(), vocab=None)
    with open(args.config_path,'r',encoding='utf-8')as f_in:
        with open(os.path.join(args.output_dir,"config.json"),'w',encoding='utf-8')as f_out:
            f_out.write(f_in.read())
    iterator = DataIterator.from_params(params.pop("iterator", vocab))
    iterator.index_with(vocab)
    trainer_params = params.pop("trainer",None)
    trainer = Trainer.from_params(model=model,
                                  serialization_dir=args.output_dir,
                                  iterator=iterator,
                                  train_data=train_dataset,
                                  validation_data=valid_dataset,
                                  params=trainer_params.duplicate())
    trainer.train()
    if test_dataset:
        logging.info("Evaluating on the test set")
        import torch
        model.load_state_dict(torch.load(os.path.join(args.output_dir,"best.th")))#"model_state_epoch_7.th")))#
        # model.load_state_dict(torch.load(os.path.join(args.output_dir,"model_state_epoch_7.th")))#"best.th")))#
        test_metrics = evaluate(model,test_dataset,iterator,
                                cuda_device=trainer_params.pop("cuda_device", 0),
                                batch_weight_key=None)
        logging.info(f"Metrics on the test set: {test_metrics}")
        with open(os.path.join(args.output_dir, "test_metrics.txt"), "w", encoding="utf-8") as f_out:
            f_out.write(f"Metrics on the test set: {test_metrics}")
    cleanup_global_logging(stdout_handler)

def evaluate(model: Model,
             instances: Iterable[Instance],
             data_iterator: DataIterator,
             cuda_device: int,
             batch_weight_key: str) -> Dict[str, Any]:
    check_for_gpu(cuda_device)
    model.eval()

    iterator = data_iterator(instances,
                                num_epochs=1,
                                shuffle=False)
    generator_tqdm = Tqdm.tqdm(iterator, total=data_iterator.get_num_batches(instances))

    # Number of batches in instances.
    batch_count = 0
    # Number of batches where the model produces a loss.
    loss_count = 0
    # Cumulative weighted loss
    total_loss = 0.0
    # Cumulative weight across all batches.
    total_weight = 0.0

    for batch in generator_tqdm:
        batch_count += 1
        batch = nn_util.move_to_device(batch, cuda_device)
        output_dict = model(**batch)
        loss = output_dict.get("loss")

        metrics = model.get_metrics()

        if loss is not None:
            loss_count += 1
            if batch_weight_key:
                weight = output_dict[batch_weight_key].item()
            else:
                weight = 1.0

            total_weight += weight
            total_loss += loss.item() * weight
            # Report the average loss so far.
            metrics["loss"] = total_loss / total_weight

        description = ', '.join(["%s: %.2f" % (name, value) for name, value
                                    in metrics.items() if not name.startswith("_")]) + " ||"
        generator_tqdm.set_description(description, refresh=False)

    final_metrics = model.get_metrics(reset=True)
    if loss_count > 0:
        # Sanity check
        if loss_count != batch_count:
            raise RuntimeError("The model you are trying to evaluate only sometimes " +
                                "produced a loss!")
        final_metrics["loss"] = total_loss / total_weight

    return final_metrics
if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument("--config_path",type=str,default="./config/profession_classifier.json")
    # arg_parser.add_argument("--config_path",type=str,default="./config/bert_profession_classifier.json")
    # arg_parser.add_argument("--config_path",type=str,default="./config/bert_gender.json")
    arg_parser.add_argument("--config_path",type=str,default="./config/fast_adv.json")
    # output_path = "./outputs/profession_classifier__1"
    # output_path = "./outputs/bert_profession_classifier7"
    output_path = "./outputs/debug32"
    # output_path = "./outputs/bert_gender"
    arg_parser.add_argument("--output_dir",type=str,default=output_path)
    args = arg_parser.parse_args()
    main(args)