from sklearn.metrics import mean_absolute_error

from scipy.stats import pearsonr


import torch
import mlflow.pytorch

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, FIFOScheduler
from ray.air import session, RunConfig
from ray.tune.integration.mlflow import mlflow_mixin
from ray.tune.integration.mlflow import MLflowLoggerCallback

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mlflow.set_tracking_uri("http://localhost:5000")
device = torch.device("cpu")
torch.manual_seed(42)

def count_parameters(model):
    # for p in model.parameters():
    #     print(p)

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def log_metrics(all_preds, all_ground_truth, all_knockout_ids, epoch, type: str, debug=False):
    mae = mean_absolute_error(all_ground_truth, all_preds)
    r2 = pearsonr(all_preds, all_ground_truth)[0]
    # all_knockout_ids = [data_augmentation_get_knockout_label(knockout_id) for knockout_id in all_knockout_ids]
    
    # k = np.array(X_test[X_test['metabolite_id'] == 'pyr']['KO_ORF'].unique())
    # mask_idx = np.argwhere(np.isin(all_knockout_ids, k)).flatten()
    # masked_mae = mean_absolute_error(all_ground_truth[mask_idx], all_preds[mask_idx])
    # masked_r2 = pearsonr(all_preds[mask_idx], all_ground_truth[mask_idx])[0]
    
    if debug:
        print(f"{mae=}")
        print(f"{r2=}")
    mlflow.log_metric(key="Mean absolute error", value=float(mae), step=epoch)
    mlflow.log_metric(key="R2 score", value=float(r2), step=epoch)
    # mlflow.log_metric(key="Masked Mean absolute error", value=float(masked_mae), step=epoch)
    # mlflow.log_metric(key="Masked R2 score", value=float(masked_r2), step=epoch)

def tune_metabolite_hyper_parameters(
    experiment_name,
    parameters,
    train_samples, 
    test_samples,
    run_fn,
    num_samples=10,
    scheduler=None,
):

    parent_id = mlflow.active_run().info.run_id if mlflow.active_run() else None

    if scheduler is None:
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=300,
            grace_period=10,
            reduction_factor=2
        )
    reporter = CLIReporter(
        metric_columns=["loss", "training_iteration"]
    )

    trainable = tune.with_parameters(
        run_fn, 
        train_samples=train_samples,
        test_samples=test_samples,
        checkpoint_dir=None,
    )

    result = tune.run(
        trainable,
        config={
            **parameters,
            "mlflow": {
                "experiment_name": experiment_name,
                "tracking_uri": mlflow.get_tracking_uri(),
                "save_artifacts": True,
                "tags": {
                    "mlflow.parentRunId": parent_id
                }
            }
        },
        num_samples=num_samples,
        scheduler=scheduler,
        # progress_reporter=reporter,
    )
    return result