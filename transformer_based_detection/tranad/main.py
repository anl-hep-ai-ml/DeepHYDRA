import os
import json
from time import time
from pprint import pprint

import pandas as pd
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from src.dataset_loader_hlt_datasets import HLTDataset
from src.dataset_loader_eclipse_datasets import EclipseDataset
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler

#from torch_profiling_utils.torchinfowriter import TorchinfoWriter
#from torch_profiling_utils.fvcorewriter import FVCoreWriter

#device='cuda:0'
device='cpu'


def _save_model_attributes(model,
                            data,
                            dataset_name):

    output_filename = f'{model.name.lower()}_{dataset_name.lower()}'

    fvcore_writer = FVCoreWriter(model, data)

    fvcore_writer.write_flops_to_json('../../evaluation/computational_intensity_analysis/data/'
                                                    f'eclipse/by_module/{output_filename}.json',
                                        'by_module')

    fvcore_writer.write_flops_to_json('../../evaluation/computational_intensity_analysis/data/'
                                                    f'eclipse/by_operator/{output_filename}.json',
                                        'by_operator')

    fvcore_writer.write_activations_to_json('../../evaluation/activation_analysis/data/'
                                                    f'eclipse/by_module/{output_filename}.json',
                                                'by_module')

    fvcore_writer.write_activations_to_json('../../evaluation/activation_analysis/data/'
                                                    f'eclipse/by_operator/{output_filename}.json',
                                                'by_operator')

    torchinfo_writer = TorchinfoWriter(model,
                                        input_data=data,
                                        verbose=0)

    torchinfo_writer.construct_model_tree()

    torchinfo_writer.show_model_tree(attr_list=['Parameters', 'MACs'])

    torchinfo_writer.get_dataframe().to_pickle('../../evaluation/parameter_analysis/'
                                                    f'data/eclipse/{output_filename}.pkl')


def _save_numpy_array(array: np.array,
                        filename: str):
    #with open(filename, 'wb') as output_file:
    #    np.save(output_file, array)
    # Create all necessary directories in the path if they don't exist
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(filename, 'wb') as output_file:
        np.save(output_file, array)


def log_gradients_in_model(model, summary_writer, step):
    for tag, value in model.named_parameters():
        if value.grad is not None:
            summary_writer.add_histogram(tag, value.cpu(), step)
            summary_writer.add_histogram(tag + "/grad", value.grad.cpu(), step)


def convert_to_windows(data, model):
    windows = []; w_size = model.n_window
    for i, g in enumerate(data): 
        if i >= w_size: w = data[i-w_size:i]
        else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
        windows.append(w if 'TranAD' in args.model or 'Attention' in args.model else w.view(-1))
    return torch.stack(windows)


def load_dataset(dataset):

    loader = []

    if 'HLT' in dataset:

        #data_source = dataset.split('_')[1]
        data_source = dataset.split('_')[0]
        variant = int(dataset.split('_')[-1])

        train_set = HLTDataset(data_source, variant,
                                        'train', False,
                                        'minmax', 'train_set_fit',
                                        applied_augmentations=\
                                                args.augmentations,
                                        augmented_dataset_size_relative=\
                                                args.augmented_dataset_size_relative,
                                        augmented_data_ratio=\
                                                args.augmented_data_ratio)

        folder = f'./checkpoints/{args.model}_{args.dataset}_{augmentation_string}_seed_{int(args.seed)}/'

        os.makedirs(folder, exist_ok=True)
        train_set.pickle_scaler(f'{folder}/scaler.pkl')

        loader.append(train_set.get_data())

        test_set = HLTDataset(data_source, variant,
                                        'test', False,
                                        'minmax', 'train_set_fit',
                                        applied_augmentations=\
                                                args.augmentations,
                                        augmented_dataset_size_relative=\
                                                args.augmented_dataset_size_relative,
                                        augmented_data_ratio=\
                                                args.augmented_data_ratio)

        loader.append(test_set.get_data())
        loader.append(test_set.get_labels())

    elif 'ECLIPSE' in dataset:

        variant = dataset.split('_')[-1].lower()

        train_set = EclipseDataset(variant, 'train', False,
                                        'minmax', 'train_set_fit',
                                        applied_augmentations=\
                                                args.augmentations,
                                        augmented_dataset_size_relative=\
                                                args.augmented_dataset_size_relative,
                                        augmented_data_ratio=\
                                                args.augmented_data_ratio)

        folder = f'./checkpoints/{args.model}_{args.dataset}_{augmentation_string}_seed_{int(args.seed)}/'

        os.makedirs(folder, exist_ok=True)
        train_set.pickle_scaler(f'{folder}/scaler.pkl')

        loader.append(train_set.get_data())

        test_set = EclipseDataset(variant, 'test', False,
                                    'minmax', 'train_set_fit',
                                    applied_augmentations=\
                                            args.augmentations,
                                    augmented_dataset_size_relative=\
                                            args.augmented_dataset_size_relative,
                                    augmented_data_ratio=\
                                            args.augmented_data_ratio)

        loader.append(test_set.get_data())
        loader.append(test_set.get_labels())


    else:

        folder = '../../datasets/smd'

        if not os.path.exists(folder):
            raise Exception('Processed Data not found.')

        for file in ['train', 'test', 'labels']:
            file = dataset + '_' + file

            loader.append(np.load(os.path.join(folder, f'{file}.npy')))

    if args.less:
        loader[0] = cut_array(0.2, loader[0])


    print(f'Train shape: {loader[0].shape}')
    print(f'Test shape: {loader[1].shape}')

    train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
    labels = loader[2]

    return train_loader, test_loader, labels


def save_model(model, optimizer, scheduler, epoch, accuracy_list):
    folder = f'checkpoints/{args.model}_{args.dataset}_{augmentation_string}_seed_{int(args.seed)}/'

    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/model.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)


def load_model(modelname, dims):
    import src.models
    model_class = getattr(src.models, modelname)
    model = model_class(dims).double()
    optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    fname = f'checkpoints/{args.model}_{args.dataset}_{augmentation_string}_seed_{int(args.seed)}/model.ckpt'

    if os.path.exists(fname) and (not args.retrain or args.test):

        print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']

    else:
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1; accuracy_list = []
    return model, optimizer, scheduler, epoch, accuracy_list


def backprop(epoch,
                model,
                data,
                dataO,
                optimizer,
                scheduler,
                training=True,
                summary_writer=None,
                dataset_name=None):

    l = nn.MSELoss(reduction = 'mean' if training else 'none')
    feats = dataO.shape[1]

    data_train_list = []
    preds_train_list = []
    data_test_list = []
    preds_test_list = []


    if 'DAGMM' in model.name:
        l = nn.MSELoss(reduction = 'none')
        compute = ComputeLoss(model, 0.1, 0.005, device, model.n_gmm)
        n = epoch + 1; w_size = model.n_window
        l1s = []; l2s = []

        data = data.to(device)

        if training:
            total_loss = 0
            for d in data:
                d = d.to(device)
                z_c, x_hat, z, gamma = model(d)
                loss = compute.forward(d.view(1, -1), x_hat.view(1, -1), z, gamma.view(1, -1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            avg_loss = total_loss / len(data)
            tqdm.write(f'Epoch {epoch},\tLoss = {avg_loss}')
            return avg_loss, optimizer.param_groups[0]['lr']
          
        else:
            anomaly_scores = []
            for d in data:
                d = d.to(device)
                z_c, x_hat, z, gamma = model(d)
                sample_energy, _ = compute.compute_energy(z, gamma)
                anomaly_scores.append(sample_energy.item())
            return anomaly_scores

        # if training:
        #     for d in data:
        #         d = d.to(device)

        #         # _save_model_attributes(model, d, dataset_name)
        #         # exit()

        #         _, x_hat, z, gamma = model(d)
        #         l1, l2 = l(x_hat, d), l(gamma, d)
        #         l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
        #         loss = torch.mean(l1) + torch.mean(l2)
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #     scheduler.step()
        #     tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
        #     return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
        # else:
        #     ae1s = []

        #     for d in data:
        #         # d = d.to(device)
        #         _, x_hat, _, _ = model(d)
        #         ae1s.append(x_hat)
        #     ae1s = torch.stack(ae1s)
        #     y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
        #     loss = l(ae1s, data)[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
        #     return loss.detach().cpu().numpy(), y_pred.detach().cpu().numpy()


    if 'Attention' in model.name:
        l = nn.MSELoss(reduction = 'none')
        n = epoch + 1; w_size = model.n_window
        l1s = []; res = []
        if training:
            for d in data:
                ae, ats = model(d)
                # res.append(torch.mean(ats, axis=0).view(-1))
                l1 = l(ae, d)
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # res = torch.stack(res); np.save('ascores.npy', res.detach().numpy())
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            ae1s, y_pred = [], []
            for d in data: 
                ae1 = model(d)
                y_pred.append(ae1[-1])
                ae1s.append(ae1)
            ae1s, y_pred = torch.stack(ae1s), torch.stack(y_pred)
            loss = torch.mean(l(ae1s, data), axis=1)
            return loss.detach().numpy(), y_pred.detach().numpy()


    elif 'OmniAnomaly' in model.name:

        model.to(device)

        if training:
            mses, klds = [], []
            for i, d in enumerate(data):
                d = d.to(device)

                # if i:
                #     _save_model_attributes(model, (d, hidden), dataset_name)
                #     exit()

                y_pred, mu, logvar, hidden = model(d, hidden if i else None)
                MSE = l(y_pred, d)
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
                loss = MSE + model.beta * KLD
                mses.append(torch.mean(MSE).item()); klds.append(model.beta * torch.mean(KLD).item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']
        else:
            y_preds = []
            for i, d in enumerate(data):
                d = d.to(device)
                y_pred, _, _, hidden = model(d, hidden if i else None)
                y_preds.append(y_pred)
            y_pred = torch.stack(y_preds)
            MSE = l(y_pred, data)
            return MSE.detach().cpu().numpy(), y_pred.detach().cpu().numpy()


    elif 'USAD' in model.name:
        l = nn.MSELoss(reduction = 'none')
        n = epoch + 1; w_size = model.n_window
        l1s, l2s = [], []

        data = data.to(device)

        if training:
            for d in data:

                # _save_model_attributes(model, d, dataset_name)
                # exit()                

                ae1s, ae2s, ae2ae1s = model(d)
                l1 = (1 / n) * l(ae1s, d) + (1 - 1/n) * l(ae2ae1s, d)
                l2 = (1 / n) * l(ae2s, d) - (1 - 1/n) * l(ae2ae1s, d)
                l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
                loss = torch.mean(l1 + l2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
            return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
        else:
            ae1s, ae2s, ae2ae1s = [], [], []

            data = data.to(device)

            for d in data:
                ae1, ae2, ae2ae1 = model(d)
                ae1s.append(ae1); ae2s.append(ae2); ae2ae1s.append(ae2ae1)
            ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
            y_pred = ae1s[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            loss = 0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)
            loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            return loss.detach().cpu().numpy(), y_pred.detach().cpu().numpy()


    elif model.name in ['GDN', 'MTAD_GAT', 'MSCRED', 'CAE_M']:
        l = nn.MSELoss(reduction = 'none')
        n = epoch + 1; w_size = model.n_window
        l1s = []

        data = data.to(device)

        if training:
            for i, d in enumerate(tqdm(data)):

                d = d.to(device)

                # _save_model_attributes(model, d, dataset_name)
                # exit()

                if 'MTAD_GAT' in model.name: 
                    x, h = model(d, h if i else None)
                else:
                    x = model(d)

                data_train_list.append(d.detach().cpu().numpy())
                preds_train_list.append(x.detach().cpu().numpy())

                # print(x.shape)

                loss = torch.mean(l(x, d))
                l1s.append(torch.mean(loss).item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')

            data_train = np.stack(data_train_list)
            preds_train = np.stack(preds_train_list)

            # _save_numpy_array(data_train, 
            #                     f'data_train_mscred_epoch_{epoch}.npy')

            # _save_numpy_array(preds_train, 
            #                     f'preds_train_mscred_epoch_{epoch}.npy')

            return np.mean(l1s), optimizer.param_groups[0]['lr']

        else:
            xs = []
            for d in tqdm(data):
                # d = d.to(device)
                if 'MTAD_GAT' in model.name: 
                    x, h = model(d, None)
                else:
                    x = model(d)

                data_test_list.append(d.detach().cpu().numpy())
                preds_test_list.append(x.detach().cpu().numpy())

                xs.append(x.detach())
            xs = torch.stack(xs)
            y_pred = xs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)   

            loss = l(xs, data.to(device))
            loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)

            data_test = np.stack(data_test_list)
            preds_test = np.stack(preds_test_list)

            # _save_numpy_array(data_test, 
            #                     f'data_test_mscred.npy')

            # _save_numpy_array(preds_test, 
            #                     f'preds_test_mscred.npy')

            return loss.detach().cpu().numpy(), y_pred.detach().cpu().numpy()


    elif model.name == 'MSCREDFull':
        l = nn.MSELoss(reduction = 'none')
        n = epoch + 1

        losses = []
        
        data = data.to(device)

        dataset = TensorDataset(data, data)
        
        bs = model.batch
        # bs = 1

        dataloader = DataLoader(dataset, batch_size=bs, drop_last=True)

        train_steps = len(dataloader)

        if training:
            for i, (d, _) in enumerate(tqdm(dataloader)):

                # d = d.to(device)

                # _save_model_attributes(model, d)
                # exit()

                x = model(d)

                data_train_list.append(d.detach().cpu().numpy())
                preds_train_list.append(x.detach().cpu().numpy())

                loss = torch.mean(l(x, d))
                losses.append(torch.mean(loss).item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                log_gradients_in_model(model,
                                        summary_writer,
                                        i + epoch*train_steps)

                summary_writer.add_scalar("Train loss",
                                            np.mean(losses),
                                            i + epoch*train_steps)

            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(losses)}')

            data_train = np.stack(data_train_list)
            preds_train = np.stack(preds_train_list)

            _save_numpy_array(data_train, 
                                f'data_train_mscred_full_epoch_{epoch}.npy')

            _save_numpy_array(preds_train, 
                                f'preds_train_mscred_full_epoch_{epoch}.npy')

            return np.mean(losses), optimizer.param_groups[0]['lr']

        else:
            xs = []
            for d, _ in tqdm(dataloader):

                x = model(d)

                data_test_list.append(d.detach().cpu().numpy())
                preds_test_list.append(x.detach().cpu().numpy())

                xs.append(x.detach())
            xs = torch.stack(xs)
            y_pred = xs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)   

            loss = l(xs, data.to(device))
            loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)

            data_test = np.stack(data_test_list)
            preds_test = np.stack(preds_test_list)

            _save_numpy_array(data_test, 
                                f'data_test_mscred_full.npy')

            _save_numpy_array(preds_test, 
                                f'preds_test_mscred_full.npy')

            return loss.detach().cpu().numpy(), y_pred.detach().cpu().numpy()


    elif 'GAN' in model.name:
        l = nn.MSELoss(reduction = 'none')
        bcel = nn.BCELoss(reduction = 'mean')
        msel = nn.MSELoss(reduction = 'mean')
        real_label, fake_label = torch.tensor([0.9]), torch.tensor([0.1]) # label smoothing
        real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
        n = epoch + 1; w_size = model.n_window
        mses, gls, dls = [], [], []
        if training:
            for d in data:
                # training discriminator
                model.discriminator.zero_grad()
                _, real, fake = model(d)
                dl = bcel(real, real_label) + bcel(fake, fake_label)
                dl.backward()
                model.generator.zero_grad()
                optimizer.step()
                # training generator
                z, _, fake = model(d)
                mse = msel(z, d) 
                gl = bcel(fake, real_label)
                tl = gl + mse
                tl.backward()
                model.discriminator.zero_grad()
                optimizer.step()
                mses.append(mse.item()); gls.append(gl.item()); dls.append(dl.item())
                # tqdm.write(f'Epoch {epoch},\tMSE = {mse},\tG = {gl},\tD = {dl}')
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
            return np.mean(gls)+np.mean(dls), optimizer.param_groups[0]['lr']
        else:
            outputs = []
            for d in data: 
                z, _, _ = model(d)
                outputs.append(z)
            outputs = torch.stack(outputs)
            y_pred = outputs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            loss = l(outputs, data)
            loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()


    elif 'TranAD' in model.name:
        l = nn.MSELoss(reduction = 'none')

        # data_x = torch.DoubleTensor(data)
        data_x = data

        dataset = TensorDataset(data_x, data_x)
        
        # bs = model.batch if training else len(data)
        bs = model.batch
        # bs = 1

        dataloader = DataLoader(dataset, batch_size=bs, drop_last=True)
        
        n = epoch + 1; w_size = model.n_window
        l1s, l2s = [], []
        
        train_steps = len(dataloader)

        if training:
            for batch_number, (d, _) in enumerate(tqdm(dataloader)):

                d = d.to(device)

                local_bs = d.shape[0]

                window = d.permute(1, 0, 2)

                elem = window[-1, :, :].view(1, local_bs, feats)

                # _save_model_attributes(model, (window, elem), dataset_name)
                # exit()

                z = model(window, elem)

                l1 = l(z, elem) if not isinstance(z, tuple)\
                        else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
                
                if isinstance(z, tuple): z = z[1]

                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)

                if summary_writer:

                    if (batch_number % 128) == 127:
                        log_gradients_in_model(model,
                                                summary_writer,
                                                batch_number +\
                                                epoch*train_steps)

                        summary_writer.add_histogram('Channel losses',
                                                        l1.detach().cpu(),
                                                        batch_number +\
                                                            epoch*train_steps)

                    summary_writer.add_scalar("Train loss",
                                                    loss,
                                                    batch_number +\
                                                        epoch*train_steps)

                optimizer.zero_grad() 
                loss.backward(retain_graph=True)
                optimizer.step()

            scheduler.step()

            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')

            return np.mean(l1s), optimizer.param_groups[0]['lr']

        else:

            zs = []
            elems = []

            bs = model.batch

            dataloader = DataLoader(dataset, batch_size=bs, drop_last=True)

            for d, _ in  tqdm(dataloader):

                d = d.to(device)

                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, bs, feats)

                elems.append(elem.detach().cpu())

                z = model(window, elem)

                if isinstance(z, tuple): z = z[1]

                zs.append(z.detach().cpu())

                torch.cuda.empty_cache()

            z = torch.cat(zs, dim=0)
            elem = torch.cat(elems, dim=0)

            z = torch.reshape(z, (1, z.shape[0]*z.shape[1], -1))
            elem = torch.reshape(elem, (1, elem.shape[0]*elem.shape[1], -1))

            loss = l(z, elem)[0]

            return loss.detach().cpu().numpy(), z.detach().cpu().numpy()[0]
    else:

        y_pred = model(data.to(device))
        loss = l(y_pred, data)

        if training:
            tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']
        else:
            return loss.detach().cpu().numpy(), y_pred.detach().cpu().numpy()


if __name__ == '__main__':

    torch.manual_seed(args.seed)

    augmentations = []

    if args.apply_augmentations:

        augmentation_string = ''
        
        for augmentation in args.augmentations:
            augmentation = augmentation.replace(' ', '')
            aug_type, factors = augmentation.split(':')

            factors = factors.split(',')

            factors_string = '_'.join(factors)

            factors = [float(factor) for factor in factors]

            if len(augmentation_string):
                augmentation_string += '_'

            augmentation_string += aug_type + '_' +\
                                        factors_string

            augmentations.append((aug_type, factors))
        
        augmentation_string += f'_rel_size_{args.augmented_dataset_size_relative}'\
                                            f'_ratio_{args.augmented_data_ratio:.2f}'
        

    if args.augmented_data_ratio == 0:
        augmentation_string = 'no_augment'

    args.augmentations = augmentations

    train_loader, test_loader, labels = load_dataset(args.dataset)
    if args.model in ['MERLIN']:
        eval(f'run_{args.model.lower()}(test_loader, labels, args.dataset)')

    model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])

    model.to(device)

    ## Prepare data

    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    trainO, testO = trainD, testD
    if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN'] or 'TranAD' in model.name: 
        trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)
    elif model.name == 'MSCREDFull':
         trainD, testD = generate_mscred_signature_matrices(trainD),\
                            generate_mscred_signature_matrices(testD)

    model.to(device)

    ### Training phase

    summary_writer = SummaryWriter()

    testD = testD.to('cpu')
    testO = testO.to('cpu')

    if not args.test:
        print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
        #num_epochs = 5
        #num_epochs = 10
        num_epochs = 1
        e = epoch + 1
        start = time()

        for e in list(range(epoch + 1, epoch + num_epochs + 1)):
            lossT, lr = backprop(e, model,
                                    trainD,
                                    trainO,
                                    optimizer,
                                    scheduler,
                                    True,
                                    summary_writer,
                                    args.dataset)

            torch.cuda.empty_cache()

            accuracy_list.append((lossT, lr))

        trainD = trainD.cpu()
        trainO = trainO.cpu()

        print(color.BOLD + 'Training time: ' + "{:10.4f}".format(time() - start) + ' s' + color.ENDC)
        save_model(model, optimizer, scheduler, e, accuracy_list)
        plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

    ### Testing phase

    torch.zero_grad = True
    model.eval()
    print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')

    testD = testD.to(device)
    testO = testO.to(device)

    loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)

    torch.cuda.empty_cache()

    testD = testD.cpu()
    testO = testO.cpu()
    # labels = labels.cpu()

    ### Plot curves

    if not args.test:
        if 'TranAD' in model.name:
            testO = torch.roll(testO, 1, 0)

        plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)

    ### Scores

    testD = testD.to(device)
    testO = testO.to(device)
    # labels = labels.to(device)

    df = pd.DataFrame()
    lossT, _ = backprop(0, model, trainD.to(device), trainO.to(device), optimizer, scheduler, training=False)

    torch.cuda.empty_cache()

    for i in range(loss.shape[1]):
        lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
        result, pred, _ = pot_eval(lt, l, ls); preds.append(pred)
        df = df.append(result, ignore_index=True)

    lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
    labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
    result, _, latencies = pot_eval(lossTfinal, lossFinal, labelsFinal)

    if 'HLT' in args.dataset:

        variant = f'{args.dataset.split("_")[-2].lower()}_'\
                                f'{args.dataset.split("_")[-1]}'

        augment_label = 'no_augment_' if augmentation_string == 'no_augment' else ''

        _save_numpy_array(lossTfinal,
                            f'../../evaluation/reduced_detection_{variant}/'\
                                            f'predictions/{args.model.lower()}_'\
                                            f'train_{augment_label}seed_{int(args.seed)}.npy')

        _save_numpy_array(lossTfinal,
                            f'../../evaluation/combined_detection_{variant}/'\
                                            f'predictions/{args.model.lower()}_'\
                                            f'train_{augment_label}seed_{int(args.seed)}.npy')

        _save_numpy_array(lossFinal,
                            f'../../evaluation/reduced_detection_{variant}/'\
                                            f'predictions/{args.model.lower()}_'\
                                            f'{augment_label}seed_{int(args.seed)}.npy')
        
    if 'ECLIPSE' in args.dataset:

        variant = args.dataset.split('_')[-1].lower()

        augment_label = 'no_augment_' if augmentation_string == 'no_augment' else ''

        _save_numpy_array(lossTfinal,
                            f'../../evaluation/reduced_detection_eclipse_{variant}/'\
                                            f'predictions/{args.model.lower()}_'\
                                            f'train_{augment_label}seed_{int(args.seed)}.npy')

        _save_numpy_array(lossTfinal,
                            f'../../evaluation/combined_detection_eclipse_{variant}/'\
                                            f'predictions/{args.model.lower()}_'\
                                            f'train_{augment_label}seed_{int(args.seed)}.npy')

        _save_numpy_array(lossFinal,
                            f'../../evaluation/reduced_detection_eclipse_{variant}/'\
                                            f'predictions/{args.model.lower()}_'\
                                            f'{augment_label}seed_{int(args.seed)}.npy')

        # parameter_dict = {"window_size": model.n_window}

        # with open(f'checkpoints/{args.model}_{args.dataset}_{augmentation_string}_seed_{int(args.seed)}/model_parameters.json', 'w') as parameter_dict_file:
        #     json.dump(parameter_dict,
        #                 parameter_dict_file)

    else:
        metrics_to_save = [int(args.seed),
                                result['ROC/AUC'],
                                result['f1'],
                                result['MCC'],
                                result['precision'],
                                result['recall']]

        metrics_to_save = np.atleast_2d(metrics_to_save)

        metrics_to_save_pd = pd.DataFrame(data=metrics_to_save)
        metrics_to_save_pd.to_csv(f'results_{args.model.lower()}_{args.dataset}.csv',
                                                                            mode='a+',
                                                                            header=False,
                                                                            index=False)


