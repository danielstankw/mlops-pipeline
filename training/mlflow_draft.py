



mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print(mlflow.get_tracking_uri())



# set an experiment, if it doesnt exist create one
mlflow.set_experiment(experiment_name='final-experiment')



# obtain a list of existing experiments
exp = mlflow.search_experiments()
for e in exp:
    print(e)
    print()




def search_for_best_model(max_results):
    """
    Search for best model X models from all the trials 
    in the experiment.
    """

    client.search_experiments()


    runs = client.search_runs(
        experiment_ids='1',
        filter_string='metrics.f1 < 0.9',
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=max_results,
        order_by=['metrics.f1 DESC'])

    for run in runs:
        print(f"Run ID: {run.info.run_id}, f1: {run.data.metrics['f1']}")

    best_model_meta_data = runs[0]

    return best_model_meta_data


def register_model():

    # Model Registry
    model_name = 'daniel1'


    mlflow.register_model(model_uri=model_uri,name=model_name)

    latest_versions = client.get_latest_versions(name=model_name)

    for lv in latest_versions:
        print(f"Version: {lv.version}, stage: {lv.current_stage}")



def transition_model_version():
    # Transition Stages: None -> Staging

    client.transition_model_version_stage(
        name=model_name,
        version=1,
        stage='Staging',
        archive_existing_versions=False
    )

    latest_versions = client.get_latest_versions(name=model_name)
    for lv in latest_versions:
        print(f"Version: {lv.version}, stage: {lv.current_stage}")




    date = datetime.today().date()
    client.update_model_version(
        name=model_name,
        version=model_version,
        description=f"The model version {model_version} was transitioned to {new_stage} on {date}"
    )


