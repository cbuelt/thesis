from metrics import *

def fill_metrics(results, model, true_parameters, pl, abc, cnn, cnn_es, alpha = 0.05, model2 = None):
    #Get mean prediction
    abc_mean = np.mean(abc, axis = 2)
    cnn_es_mean = np.mean(cnn_es, axis = 2)
    # PL
    mse = get_mse(true_parameters, pl, sd = True)
    imse = get_integrated_error(model, true = true_parameters, estimate = pl, sd = True, model2 = model2)
    kld = get_integrated_kld(true_parameters, model, pl, sd = True)
    results.loc[(model, "MSE_r"), "PL"] = f"{mse[0][0]:.4f} ({mse[1][0]:.2f})"
    results.loc[(model, "MSE_s"), "PL"] = f"{mse[0][1]:.4f} ({mse[1][1]:.2f})"
    results.loc[(model, "MSE_ext"), "PL"] = f"{imse[0]:.4f} ({imse[1]:.2f})"
    results.loc[(model, "KL"), "PL"] = f"{kld[0]:.4f} ({kld[1]:.2f})"

    # CNN 
    mse = get_mse(true_parameters, cnn, sd = True)
    imse = get_integrated_error(model, true = true_parameters, estimate = cnn, sd = True, model2 = model2)
    kld = get_integrated_kld(true_parameters, model, cnn, sd = True)
    results.loc[(model, "MSE_r"), "CNN"] = f"{mse[0][0]:.4f} ({mse[1][0]:.2f})"
    results.loc[(model, "MSE_s"), "CNN"] = f"{mse[0][1]:.4f} ({mse[1][1]:.2f})"
    results.loc[(model, "MSE_ext"), "CNN"] = f"{imse[0]:.4f} ({imse[1]:.2f})"
    results.loc[(model, "KL"), "CNN"] = f"{kld[0]:.4f} ({kld[1]:.2f})"

    # ABC
    mse = get_mse(true_parameters, abc_mean, sd = True)
    imse = get_integrated_error(model, true = true_parameters, estimate = abc, method = "sample", sd = True, model2 = model2)
    kld = get_integrated_kld(true_parameters, model, abc_mean, sd = True)
    results.loc[(model, "MSE_r"), "ABC"] = f"{mse[0][0]:.4f} ({mse[1][0]:.2f})"
    results.loc[(model, "MSE_s"), "ABC"] = f"{mse[0][1]:.4f} ({mse[1][1]:.2f})"
    results.loc[(model, "MSE_ext"), "ABC"] = f"{imse[0]:.4f} ({imse[1]:.2f})"
    results.loc[(model, "KL"), "ABC"] = f"{kld[0]:.4f} ({kld[1]:.2f})"

    quantiles = np.quantile(abc, [alpha/2,1-(alpha/2)], axis = 2)
    iscore = get_interval_score(true_parameters, alpha = alpha, q_left = quantiles[0], q_right = quantiles[1], sd = True)
    iis = get_integrated_is(model, true = true_parameters, estimate = abc, alpha = alpha, sd = True, model2 = model2)
    es = get_energy_score(true_parameters,abc, sd = True)
    results.loc[(model, "IS_r",), "ABC"] = f"{iscore[0][0]:.4f} ({iscore[1][0]:.2f})"
    results.loc[(model, "IS_s",), "ABC"] = f"{iscore[0][1]:.4f} ({iscore[1][1]:.2f})"
    results.loc[(model, ["IIS"]), "ABC"] = f"{iis[0]:.4f} ({iis[1]:.2f})"
    results.loc[(model, ["ES"]), "ABC"] = f"{es[0]:.4f} ({es[1]:.2f})"

    # CNN ES
    mse = get_mse(true_parameters, cnn_es_mean, sd = True)
    imse = get_integrated_error(model, true = true_parameters, estimate = cnn_es, method = "sample", sd = True, model2 = model2)
    kld = get_integrated_kld(true_parameters, model, cnn_es_mean, sd = True)
    results.loc[(model, "MSE_r"), "CNN_ES"] = f"{mse[0][0]:.4f} ({mse[1][0]:.2f})"
    results.loc[(model, "MSE_s"), "CNN_ES"] = f"{mse[0][1]:.4f} ({mse[1][1]:.2f})"
    results.loc[(model, "MSE_ext"), "CNN_ES"] = f"{imse[0]:.4f} ({imse[1]:.2f})"
    results.loc[(model, "KL"), "CNN_ES"] = f"{kld[0]:.4f} ({kld[1]:.2f})"

    quantiles = np.quantile(cnn_es, [alpha/2,1-(alpha/2)], axis = 2)
    iscore = get_interval_score(true_parameters, alpha = alpha, q_left = quantiles[0], q_right = quantiles[1], sd = True)
    iis = get_integrated_is(model, true = true_parameters, estimate = cnn_es, alpha = alpha, sd = True, model2 = model2)
    es = get_energy_score(true_parameters, cnn_es, sd = True)
    results.loc[(model, "IS_r",), "CNN_ES"] = f"{iscore[0][0]:.4f} ({iscore[1][0]:.2f})"
    results.loc[(model, "IS_s",), "CNN_ES"] = f"{iscore[0][1]:.4f} ({iscore[1][1]:.2f})"
    results.loc[(model, ["IIS"]), "CNN_ES"] = f"{iis[0]:.4f} ({iis[1]:.2f})"
    results.loc[(model, ["ES"]), "CNN_ES"] = f"{es[0]:.4f} ({es[1]:.2f})"
    return results


def load_predictions(data_path, results_path, model):
    # Load true parameters
    true_parameters = pyreadr.read_r(data_path+model+"_test_params.RData")["test_params"].to_numpy()[0:n_test]
    # Load PL
    pl = pyreadr.read_r(results_path+model+"_pl.RData")["results"].to_numpy()[0:n_test,0:2]
    # Load ABC
    abc = xr.open_dataset(results_path + model + "_abc_results.nc").results.data[0:n_test,0:2]
    # Load normal network
    cnn = np.load(results_path+model+"_cnn.npy")[0:n_test]
    # Load energy network
    cnn_es = np.load(results_path+model+"_cnn_es.npy")[0:n_test]

    return true_parameters, pl, abc, cnn, cnn_es

def get_results_table(exp, models, model2 = None, save = True):
    # Set data paths
    data_path = f'data/{exp}/data/'
    results_path = f'data/{exp}/results/'
    # Set filename
    filename = models[0] + "_results_table.pkl" if exp == "outside_model" else "results_table.pkl"

    # Prepare dataframe
    metrics = ["MSE_r", "MSE_s", "MSE_ext", "KL", "IS_r", "IS_s", "IIS", "ES"]
    results = pd.DataFrame("-", index = pd.MultiIndex.from_product([models, metrics]), columns = ["PL", "CNN", "ABC", "CNN_ES"])
    
    for model in models:
        true_parameters, pl, abc, cnn, cnn_es = load_predictions(data_path, results_path, model)
        results = fill_metrics(results, model, true_parameters, pl, abc, cnn, cnn_es, model2 = model2)

    if save:
        results.to_pickle(results_path + filename)
        print(f"Successfully saved table at {results_path + filename}")
    else:
        return results


if __name__ == "__main__":
    n_test = 5

    # Normal predictions
    exp = "normal"
    get_results_table(exp, models = ["brown", "powexp"])

    # Outside parameters
    exp = "outside_parameters"
    get_results_table(exp, models = ["brown"])

    # Outside model - Whitmat
    exp = "outside_model"
    get_results_table(exp, models = ["whitmat"], model2 = "powexp")

    # Outside model - Smith
    get_results_table(exp, models = ["brown"])





    