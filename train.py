import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from NeuTucF import NeuTucFEngine
from SANeuTF import SANeuTFEngine
from data import load



NeuTucF_config = {
    "model": "saneutf",
    "num_epoch": 300,
    "batch_size": 1024,
    "optimizer": "adam",
    "learning_rate": 4e-3,
    "dimensions": [142, 4500, 64],
    "embedding_dims": [5, 5, 5],
    "l2_regularization": 0,
    "tune_list": [
        [1e-3, 2e-3, 4e-3, 6e-3, 8e-3, 1e-2], 
        [0, 1e-10, 1e-8], 
    ],
    "flag_bias": False,
    "use_mps": False,
    "pretrain": False,
    "model_dir": "checkpoints/",
}

QoS_densities = [2.0]
QoS_types = ["rt", "tp"]

# Network_densities = [10.0]
# Network_types = ["7a_4"]

# Guangzhou_densities = [20.0]
# Guangzhou_types = ["gzspeed"]

# Hangzhou_densities = [1.0]
# Hangzhou_types = ["hzflow"]

# Seattle_densities = [1.0]
# Seattle_types = ["Speed"]

NYCTaxi_densities = [20.0]
NYCTaxi_types = ["NYCTaxi"]

PCTemp_densities = [10.0]
PCTemp_types = ["PCTemp"]

if __name__ == "__main__":
    dataset = "PCTemp"
    model_config = NeuTucF_config
    flag_tune = False  # Tune the hyperparameters or not
    flag_log = True  # Transform the data with log or not, false for PCTemp
    flag_norm = True  # Normalize the data or not
    flag_training_summary = True  # Log the training summary or not

    types = []
    densities = []

    if dataset == "WSDREAM":
        DIR = "/workspaces/01_project/SNeuTucF/NeuTucF/dataset/"
        densities = QoS_densities
        types = QoS_types
    # elif dataset == "Network":
    #     DIR = "/Users/matthew/Documents/Projects/Data/NetworkData/"
    #     densities = Network_densities
    #     types = Network_types
    # elif dataset == "Guangzhou":
    #     DIR = "/Users/matthew/Documents/Projects/Data/GuangZhou/"
    #     densities = Guangzhou_densities
    #     types = Guangzhou_types
    # elif dataset == "Hangzhou":
    #     DIR = "/Users/matthew/Documents/Projects/Data/Hangzhou/"
    #     densities = Hangzhou_densities
    #     types = Hangzhou_types
    # elif dataset == "Seattle":
    #     DIR = "/Users/matthew/Documents/Projects/Data/Seattle/"
    #     densities = Seattle_densities
    #     types = Seattle_types
    elif dataset == "NYCTaxi":
        DIR = "/workspaces/01_project/SNeuTucF/NeuTucF/dataset/"
        densities = NYCTaxi_densities
        types = NYCTaxi_types
    elif dataset == "PCTemp":
        DIR = "/workspaces/01_project/SNeuTucF/NeuTucF/dataset/"
        densities = PCTemp_densities
        types = PCTemp_types
    elif dataset == "Test":
        DIR = "/Users/matthew/Documents/Projects/Data/NoCongestion/"
        densities = [2.0]
        types = ["rt", "tp"]

    MODEL_TYPE = model_config["model"]
    N_ROUNDS = 5
    TOLERANCE = 1e-5  # Tolerance for early stopping
    N_CVG = 5  # Number of consecutive epochs for convergence
    MIN_EPOCHS = 20  # Minimum number of rounds for training

    ranks = model_config["embedding_dims"]

    train_filepath = ""
    valid_filepath = ""
    test_filepath = ""
    for data_type in types:
        for den in densities:
            print(f"Loading data for density {den:.1f} ...")
            if dataset == "WSDREAM" or dataset == "Test":
                train_filepath = (
                    DIR + f"{data_type}Rate_nozero_p{den:.1f}_tr{den:.1f}.txt"
                )
                valid_filepath = (
                    DIR + f"{data_type}Rate_nozero_p{den:.1f}_va{den:.1f}.txt"
                )
                test_filepath = (
                    DIR + f"{data_type}Rate_nozero_p{den:.1f}_te{den:.1f}.txt"
                )
            elif (
                dataset == "Network"
                or dataset == "Guangzhou"
                or dataset == "Hangzhou"
                or dataset == "Seattle"
                or dataset == "NYCTaxi"
                or dataset == "PCTemp"
            ):
                train_filepath = (
                    DIR + f"{data_type}_p{den:.1f}_tr{den:.1f}.csv"
                )
                valid_filepath = (
                    DIR + f"{data_type}_p{den:.1f}_va{den:.1f}.csv"
                )
                test_filepath = DIR + f"{data_type}_p{den:.1f}_te{den:.1f}.csv"

            train_dataset = load(train_filepath)
            valid_dataset = load(valid_filepath)
            test_dataset = load(test_filepath)

            min_val = torch.min(train_dataset.values)
            max_val = torch.max(train_dataset.values)

            max_A_index = torch.max(train_dataset.A_Indices) + 1
            max_B_index = torch.max(train_dataset.B_Indices) + 1
            max_C_index = torch.max(train_dataset.C_Indices) + 1

            config = model_config
            config["dimensions"] = [max_A_index, max_B_index, max_C_index]

            # Dataloader for training and testing
            data_loader = DataLoader(
                train_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
            )
            eval_loader = DataLoader(
                valid_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
            )

            if flag_tune:
                base_dir = f"Tuning_results/{dataset}/{data_type}/d{den:.1f}/"
            else:
                base_dir = f"Results/{dataset}/{data_type}/d{den:.1f}/"

            if isinstance(ranks, list):
                rank_info = f"_".join([str(r) for r in ranks])
                result_dir = base_dir + f"{MODEL_TYPE}/Ranks{rank_info}/"
            else:
                result_dir = base_dir + f"{MODEL_TYPE}/Ranks{ranks}/"

            Path(result_dir).mkdir(parents=True, exist_ok=True)
            model_dir = result_dir + "models/"
            Path(model_dir).mkdir(parents=True, exist_ok=True)

            if flag_tune:
                summary_file = result_dir + "tune_summary.txt"
            else:
                op_type = config["optimizer"]
                eta = config["learning_rate"]
                lamb = config["l2_regularization"]
                result_dir = (
                    result_dir + f"{op_type}_lr{eta:.1e}_reg{lamb:.1e}/"
                )
                Path(result_dir).mkdir(parents=True, exist_ok=True)
                summary_file = result_dir + "summary.txt"
            with open(summary_file, "w", encoding="utf-8") as f:
                if flag_tune:
                    f.write(
                        "eta lambda MAE MAE_Epoch MRE MRE_Epoch RMSE RMSE_Epoch\n"
                    )
                    para_lists = config["tune_list"]

                    eta_best = 0
                    lamb_best = 0
                    rmse_min = np.inf
                    for eta in para_lists[0]:
                        for lamb in para_lists[1]:
                            config["learning_rate"] = eta
                            config["l2_regularization"] = lamb

                            # Ensure reproducibility for each run
                            torch.manual_seed(0)

                            engine = None
                            # if MODEL_TYPE == "mlp":
                            #     engine = MLPEngine(
                            #         config,
                            #         [min_val, max_val],
                            #         flag_norm,
                            #         flag_log,
                            #     )
                            # elif MODEL_TYPE == "n3cp":
                            #     engine = NCPFEngine(config)
                            # elif MODEL_TYPE == "neutf":
                            #     engine = NeuTFEngine(
                            #         config,
                            #         [min_val, max_val],
                            #         flag_norm,
                            #         flag_log,
                            #     )
                            if MODEL_TYPE == "neutucf":
                                engine = NeuTucFEngine(
                                    config,
                                    [min_val, max_val],
                                    flag_norm,
                                    flag_log,
                                )
                            elif MODEL_TYPE == "saneutf":  # <--- 新增这部分
                                engine = SANeuTFEngine(
                                    config,[min_val, max_val],
                                    flag_norm,
                                    flag_log,
                                )
                            # elif MODEL_TYPE == "neutucfae":
                            #     engine = NeuTucFAEEngine(
                            #         config,
                            #         [min_val, max_val],
                            #         flag_norm,
                            #         flag_log,
                            #     )
                            # elif MODEL_TYPE == "neutucfvae":
                            #     engine = NeuTucFVAEEngine(
                            #         config,
                            #         [min_val, max_val],
                            #         flag_norm,
                            #         flag_log,
                            #     )

                            mae_best = np.inf
                            mre_best = np.inf
                            rmse_best = np.inf

                            mae_pre = np.inf
                            mre_pre = np.inf
                            rmse_pre = np.inf

                            mae_best_epoch = 0
                            mre_best_epoch = 0
                            rmse_best_epoch = 0

                            flag_rmse = False
                            flag_mae = False
                            flag_mre = False

                            mae, mape, rmse = engine.evaluate(eval_loader)
                            print(
                                f"[Training Epoch 0]: {mae:.4f},"
                                f" {mape:.4f}, {rmse:.6f}, 0"
                            )

                            for epoch in range(config["num_epoch"]):
                                start_time = time.time()
                                engine.train_an_epoch(
                                    data_loader, epoch_id=epoch
                                )
                                mae, mape, rmse = engine.evaluate(eval_loader, epoch_id=epoch)
                                if epoch % 1 == 0:
                                    end_time = time.time()
                                    print(
                                        f"[Training Epoch {epoch+1}]: {mae:.4f},"
                                        f" {mape:.4f}, {rmse:.6f},"
                                        f" {end_time - start_time}"
                                    )
                                    start_time = end_time

                                if (
                                    rmse_pre - rmse > TOLERANCE
                                    and rmse < rmse_best
                                ):
                                    rmse_best = rmse
                                    rmse_best_epoch = epoch
                                    model_file = model_dir + f"best_rmse.model"
                                    engine.save(model_file)
                                elif epoch - rmse_best_epoch > N_CVG:
                                    flag_rmse = True
                                    if (
                                        flag_mae
                                        and flag_mre
                                        and epoch > MIN_EPOCHS
                                    ):
                                        break

                                if (
                                    mae_pre - mae > TOLERANCE
                                    and mae < mae_best
                                ):
                                    mae_best = mae
                                    mae_best_epoch = epoch
                                    model_file = model_dir + f"best_mae.model"
                                    engine.save(model_file)
                                elif epoch - mae_best_epoch > N_CVG:
                                    flag_mae = True
                                    if (
                                        flag_rmse
                                        and flag_mre
                                        and epoch > MIN_EPOCHS
                                    ):
                                        break

                                if (
                                    mre_pre - mape > TOLERANCE
                                    and mape < mre_best
                                ):
                                    mre_best = mape
                                    mre_best_epoch = epoch
                                    model_file = model_dir + f"best_mre.model"
                                    engine.save(model_file)
                                elif epoch - mre_best_epoch > N_CVG:
                                    flag_mre = True
                                    if (
                                        flag_rmse
                                        and flag_mae
                                        and epoch > MIN_EPOCHS
                                    ):
                                        break

                            model_file = model_dir + f"best_mae.model"
                            engine.model.load_model(model_file)
                            mae, _, _ = engine.pred(test_loader)

                            model_file = model_dir + f"best_mre.model"
                            engine.model.load_model(model_file)
                            _, mre, _ = engine.pred(test_loader)

                            model_file = model_dir + f"best_rmse.model"
                            engine.model.load_model(model_file)
                            _, _, rmse = engine.pred(test_loader)

                            print(
                                f"ETA-{eta:.1e}, LAMBDA-{lamb:.1e} "
                                f"MAE:{mae:.6f}, MAE_Epoch:{mae_best_epoch} "
                                f"MRE:{mre:.6f}, MRE_Epoch:{mre_best_epoch} "
                                f"RMSE:{rmse:.6f}, RMSE_Epoch:{rmse_best_epoch}"
                            )

                            f.write(
                                f"{eta:.1e} {lamb:.1e} "
                                f"{mae:.6f} {mae_best_epoch} "
                                f"{mre:.6f} {mre_best_epoch} "
                                f"{rmse:.6f} {rmse_best_epoch}\n"
                            )
                            f.flush()
                    f.close()
                else:
                    f.write("MAE MAE_Epoch MRE MRE_Epoch RMSE RMSE_Epoch\n")
                    for rnd in range(N_ROUNDS):
                        # if MODEL_TYPE == "mlp":
                        #     engine = MLPEngine(
                        #         config, [min_val, max_val], flag_norm, flag_log
                        #     )
                        # elif MODEL_TYPE == "neutf":
                        #     engine = NeuTFEngine(
                        #         config, [min_val, max_val], flag_norm, flag_log
                        #     )
                        # elif MODEL_TYPE == "n3cp":
                        #     engine = NCPFEngine(config)
                        if MODEL_TYPE == "neutucf":
                            engine = NeuTucFEngine(
                                config, [min_val, max_val], flag_norm, flag_log
                            )
                        elif MODEL_TYPE == "saneutf":     # <--- 新增这部分
                            engine = SANeuTFEngine(
                                config, [min_val, max_val], flag_norm, flag_log
                            )
                        # elif MODEL_TYPE == "aetucd":
                        #     engine = AETucDEngine(
                        #         config, [min_val, max_val], flag_norm, flag_log
                        #     )
                        # elif MODEL_TYPE == "neutucfae":
                        #     engine = NeuTucFAEEngine(
                        #         config, [min_val, max_val], flag_norm, flag_log
                        #     )
                        # elif MODEL_TYPE == "neutucfae_bce":
                        #     engine = NeuTucFAEBEngine(
                        #         config, [min_val, max_val], flag_norm, flag_log
                        #     )
                        # elif MODEL_TYPE == "neutucfvae":
                        #     engine = NeuTucFVAEEngine(
                        #         config, [min_val, max_val], flag_norm, flag_log
                        #     )
                        # elif MODEL_TYPE == "neutucfmlp":
                        #     engine = NeuTucFMLPEngine(
                        #         config, [min_val, max_val], flag_norm, flag_log
                        #     )

                        mae_best = np.inf
                        mre_best = np.inf
                        rmse_best = np.inf

                        mae_pre = np.inf
                        mre_pre = np.inf
                        rmse_pre = np.inf

                        mae_best_epoch = 0
                        mre_best_epoch = 0
                        rmse_best_epoch = 0

                        flag_rmse = False
                        flag_mae = False
                        flag_mre = False

                        mae, mape, rmse = engine.evaluate(eval_loader)
                        print(
                            f"[Training Epoch 0]: {mae:.4f},"
                            f" {mape:.4f}, {rmse:.6f}, 0"
                        )

                        for epoch in range(config["num_epoch"]):
                            start_time = time.time()
                            engine.train_an_epoch(data_loader, epoch_id=epoch)
                            end_time = time.time()

                            mae, mape, rmse = engine.evaluate(eval_loader, epoch_id=epoch)
                            valid_end_time = time.time()

                            if epoch % 1 == 0:
                                print(
                                    f"[Training Epoch {epoch+1}]: {mae:.4f},"
                                    f" {mape:.4f}, {rmse:.6f},"
                                    f" {end_time - start_time},"
                                    f" {valid_end_time - end_time}"
                                )

                            if (
                                rmse_pre - rmse > TOLERANCE
                                and rmse < rmse_best
                            ):
                                rmse_best = rmse
                                rmse_best_epoch = epoch
                                model_file = model_dir + f"best_rmse.model"
                                engine.save(model_file)
                            elif epoch - rmse_best_epoch > N_CVG:
                                flag_rmse = True
                                if (
                                    flag_mae
                                    and flag_mre
                                    and epoch > MIN_EPOCHS
                                ):
                                    break

                            if mae_pre - mae > TOLERANCE and mae < mae_best:
                                mae_best = mae
                                mae_best_epoch = epoch
                                model_file = model_dir + f"best_mae.model"
                                engine.save(model_file)
                            elif epoch - mae_best_epoch > N_CVG:
                                flag_mae = True
                                if (
                                    flag_rmse
                                    and flag_mre
                                    and epoch > MIN_EPOCHS
                                ):
                                    break

                            if mre_pre - mape > TOLERANCE and mape < mre_best:
                                mre_best = mape
                                mre_best_epoch = epoch
                                model_file = model_dir + f"best_mre.model"
                                engine.save(model_file)
                            elif epoch - mre_best_epoch > N_CVG:
                                flag_mre = True
                                if (
                                    flag_rmse
                                    and flag_mae
                                    and epoch > MIN_EPOCHS
                                ):
                                    break

                        start_time = time.time()
                        model_file = model_dir + f"best_mae.model"
                        engine.model.load_model(model_file)
                        mae, _, _ = engine.pred(test_loader)

                        model_file = model_dir + f"best_mre.model"
                        engine.model.load_model(model_file)
                        _, mre, _ = engine.pred(test_loader)

                        model_file = model_dir + f"best_rmse.model"
                        engine.model.load_model(model_file)
                        _, _, rmse = engine.pred(test_loader)
                        end_time = time.time()

                        print(
                            f"Density-{den:.1f}, Round-{rnd+1:2d}, "
                            f"MAE:{mae:.6f}, MAE_Epoch:{mae_best_epoch}, "
                            f"MRE:{mre:.6f}, MRE_Epoch:{mre_best_epoch}, "
                            f"RMSE:{rmse:.6f}, RMSE_Epoch:{rmse_best_epoch}, "
                            f"Test time:{end_time-start_time:.2f}"
                        )

                        f.write(
                            f"{mae:.4f} {mae_best_epoch} "
                            f"{mre:.4f} {mre_best_epoch} "
                            f"{rmse:.4f} {rmse_best_epoch}\n"
                        )
                        f.flush()
                    f.close()
