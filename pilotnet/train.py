import os

import dicto as do
import fire
import tensorflow as tf

from . import estimator as est
from . import data_pipelines

PROJECT_DIR = os.path.dirname(__file__)
CONFIGS_FILEPATH = os.path.join(PROJECT_DIR, "configs/train.yml")

@do.fire_options(CONFIGS_FILEPATH)
def main(data_dir, job_dir, params):

    params = do.Dicto(params)



    tf.logging.set_verbosity(tf.logging.DEBUG)
    print("job_dir", job_dir)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    run_config = tf.estimator.RunConfig(
        model_dir=job_dir,
        save_summary_steps=params.summary_steps,
        save_checkpoints_steps=params.save_checkpoints_steps,
        session_config=config
    )



    estimator = tf.estimator.Estimator(
        model_fn=est.model_fn,
        params=params,
        config=run_config,
    )

    exporter = tf.estimator.LatestExporter(
        params.project,
        lambda: data_pipelines.serving_input_fn(params),
    )

    train_spec = tf.estimator.TrainSpec(
        lambda: data_pipelines.input_fn(data_dir, params),
        max_steps=params.max_steps,
    )

    test_spec = tf.estimator.EvalSpec(
        lambda: data_pipelines.input_fn(data_dir, params),
        steps=params.eval_steps,
        exporters=[exporter], 
    )

    print("Start training and evaluate")

    tf.estimator.train_and_evaluate(estimator, train_spec, test_spec)

    # for _ in range(1000000):

    #     tf.estimator.train_and_evaluate(estimator, train_spec, test_spec)


    print("exporting")

    estimator.export_savedmodel(
        os.path.join(job_dir, "export", params.project),
        lambda: data_pipelines.serving_input_fn(params)
    )

if __name__ == "__main__":
    fire.Fire(main)