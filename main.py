import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from utils.logger import LOGGER
from trainer import Metrics, Trainer
from datautils.dataloader import prepare_loaders
from rich.progress import BarColumn, Progress, TimeRemainingColumn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from zipfile import ZipFile
from utils.logger import add_log_to_file

def main(args):
    config = args.__dict__
    config = Trainer.check_config(config)
    
    prepare_loaders(config)
    
    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}% [progress.completed]{task.completed}",
        TimeRemainingColumn(),
    )

    with progress:
        trainer = Trainer(config, progress)

        trainer.load_model()
        if args.task=='train':
            assert config['train_loader'], "Train dataset must exist!"
            trainer.do_train()    

        if args.task=='eval':
            assert config['test_loader'], "Test dataset must exist!"
            
            formatted_date = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            add_log_to_file(Path(config['model_file']).parent / f"eval_{formatted_date}.log")        

            LOGGER.info("\n\n" + "="*100 + "\n\t\t\t\t\t Evaluating Network\n" + "="*100)

            trainer.print_config()
            labels, preds, loss = trainer.eval_model('test_loader')    

            labels = np.argmax(labels, axis=1) if len(labels[-1])>1 else labels
            preds = np.argmax(preds, axis=1) if len(preds[-1])>1 else preds
            if config['nr_classes']<=2:
                metric = Metrics(
                    train_loss=None,
                    train_acc=None,
                    train_prec=None,
                    train_recall=None,
                    train_f1=None,
                    valid_loss=loss,
                    valid_acc=accuracy_score(labels, preds),
                    valid_prec=precision_score(labels, preds),
                    valid_recall=recall_score(labels, preds),
                    valid_f1=f1_score(labels, preds),
                )
            else:
                metric = Metrics(
                    train_loss=None,
                    train_acc=None,
                    train_prec=None,
                    train_recall=None,
                    train_f1=None,
                    valid_loss=loss,
                    valid_acc=accuracy_score(labels, preds),
                    valid_prec=precision_score(labels, preds, average='weighted'),
                    valid_recall=recall_score(labels, preds, average='weighted'),
                    valid_f1=f1_score(labels, preds, average='weighted'),
                )

            LOGGER.info("-"*30)
            LOGGER.info("Report")
            LOGGER.info("="*30)
            LOGGER.info(metric)
            LOGGER.info("\n\n")

        if args.task=='predict':
            assert config['test_loader'], "Test dataset must exist!"
            formatted_date = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            add_log_to_file(Path(config['model_file']).parent / f"predict_{formatted_date}.log")        

            LOGGER.info("\n\n" + "="*100 + "\n\t\t\t\t\t Predicting Network\n" + "="*100)

            output_file = Path(config['model_file']).parent / f"prediction_{formatted_date}.csv"
            raw_output_file = Path(config['model_file']).parent / f"raw_prediction_{formatted_date}.csv"
            LOGGER.info(f"\t\t Output file = {output_file}")
            LOGGER.info(f"\t\t Raw Output file = {raw_output_file}")

            file_ids, preds, probs = trainer.predict_model()

            df_raw = pd.DataFrame(file_ids, columns=["filename"])
            df_raw["filename"] = df_raw["filename"].apply(lambda x: str(x)+'.jpg')

            LOGGER.info("\n\n" + "-"*30 + "\n\t\t\t\t\t Statistics \n" + "-"*30)
            LOGGER.info(f"Total Records {len(df_raw)}")
            if config['nr_classes']>2:
                # multiclass (Task B)
                df_raw['misogynous'] = np.max(preds, axis=1)
                df_raw['misogynous'] = df_raw['misogynous'].astype(int)
                if config['nr_classes']==4:
                    df_raw = pd.concat([df_raw, pd.DataFrame(preds, columns=["shaming","stereotype","objectification","violence"])], axis=1)
                    df_raw = pd.concat([df_raw, pd.DataFrame(probs, columns=["shaming_prob","stereotype_prob",
                                                                            "objectification_prob","violence_prob"])], axis=1)
                    # since we check only positives, we take the highest prob for non detected
                    def get_best_prob(rec):
                        best = rec[["shaming_prob","stereotype_prob","objectification_prob","violence_prob"]].idxmax()    
                        rec[best.replace("_prob","")] = 1
                        return rec
                    df_raw[df_raw['misogynous']==0] =  df_raw[df_raw['misogynous']==0].apply(get_best_prob)
                else:
                    df_raw = pd.concat([df_raw, pd.DataFrame(preds, columns=["non_misogynous","shaming","stereotype","objectification","violence"])], axis=1)
                    df_raw = pd.concat([df_raw, pd.DataFrame(probs, columns=["non_misogynous_prob","shaming_prob","stereotype_prob",
                                                                            "objectification_prob","violence_prob"])], axis=1)
                df_raw['misogynous'] = df_raw[["shaming","stereotype","objectification","violence"]].max(axis=1)

                if config['nr_classes']==4:
                    df_raw_negatives = pd.read_csv(Path(config['data_path']) / ("test.csv"), sep="\t")
                    df_raw_negatives = df_raw_negatives[df_raw_negatives.misogynous==0]
                    df_raw_negatives['pred_shaming'] = 0
                    df_raw_negatives['pred_stereotype'] = 0
                    df_raw_negatives['pred_objectification'] = 0
                    df_raw_negatives['pred_violence'] = 0
                    df_raw = pd.concat([df_raw, df_raw_negatives])

                df_submit = df_raw[["filename", "misogynous","shaming","stereotype","objectification","violence"]]
                df_submit['misogynous'] = df_submit['misogynous'].astype(int)
                df_submit['shaming'] = df_submit['shaming'].astype(int)
                df_submit['stereotype'] = df_submit['stereotype'].astype(int)
                df_submit['objectification'] = df_submit['objectification'].astype(int)
                df_submit['violence'] = df_submit['violence'].astype(int)

                LOGGER.info("\nMisogynous:")
                LOGGER.info(df_submit.misogynous.value_counts())
                LOGGER.info("\nShaming:")
                LOGGER.info(df_submit.shaming.value_counts())
                LOGGER.info("\nStereotype:")
                LOGGER.info(df_submit.stereotype.value_counts())
                LOGGER.info("\nObjectification:")
                LOGGER.info(df_submit.objectification.value_counts())
                LOGGER.info("\nViolence:")
                LOGGER.info(df_submit.violence.value_counts())

            else:
                # binary (Task A)
                df_raw['misogynous'] = np.argmax(preds, axis=1)
                df_raw['misogynous'] = df_raw['misogynous'].astype(int)
                df_raw = pd.concat([df_raw, pd.DataFrame(probs, columns=["misogynous_prob_0","misogynous_prob_1"])], axis=1)
                df_submit = df_raw[["filename", "misogynous"]]
                LOGGER.info("\nMisogynous:")
                LOGGER.info(df_submit.misogynous.value_counts())

            df_submit.to_csv(output_file, sep="\t", index=None, header=None)
            df_raw.to_csv(raw_output_file, sep="\t", index=None)

            zip_name = output_file.with_suffix(".zip")
            with ZipFile(zip_name,'w') as zip:
                zip.write(output_file, "answer.txt")
            




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    Trainer.prepare_args(parser)

    args, _ = parser.parse_known_args()
    main(args)
