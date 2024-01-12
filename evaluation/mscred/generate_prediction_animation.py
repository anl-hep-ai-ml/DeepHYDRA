# Copyright (C) 2023 CERN for the benefit of the ATLAS collaboration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import numpy as np
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import animation
import seaborn as sns
from tqdm import tqdm


writer_type = 'ffmpeg'
codec = 'mpeg4'


def load_numpy_array(filename: str):
    with open(filename, 'rb') as output_file:
        return np.load(output_file)


def render_autocorr_data_pred(data_all: np.array,
                                preds_all: np.array,
                                output_filename: str,
                                artist_name: str,
                                fps: int = 60,
                                interval: int = 1,
                                fig_size: tuple = (19.2, 5.4),
                                label_size: float = 14,
                                title_size: float = 20,
                                cmap = 'viridis',
                                fig_facecolor = 'darkgrey',
                                ax_facecolor = 'lightgrey',
                                seaborn_plotting_context: str = 'talk',
                                writer_args: list = ['-qscale:v', '3']) -> None:


    sns.set(rc={'figure.figsize': fig_size,
                    'figure.titlesize': title_size,
                    'axes.labelsize': label_size,
                    'axes.titlesize': title_size,
                    'figure.facecolor': fig_facecolor,
                    'axes.facecolor': ax_facecolor})

    pbar = tqdm(total=len(data_all),
                    desc='Rendering data/prediction/diff matrices')

    fig = plt.figure()

    def animate(position):

        plt.clf()

        ax_data, ax_preds, ax_diff = fig.subplots(1, 3)

        with sns.plotting_context(seaborn_plotting_context):

            data = data_all[position]
            pred = preds_all[position]

            diff = np.abs(data - pred)/(data + 1e-6)

            # Data

            shape_original = data.shape

            data_scaler = MinMaxScaler()

            data = np.atleast_2d(data.reshape(-1)).T
            data = data_scaler.fit_transform(data).T
            data = data.reshape(*shape_original)

            matrix_dim = int(math.sqrt(data.shape[0]))

            data = data.reshape(matrix_dim, matrix_dim)

            ax_data.tick_params(labelbottom=False, labelleft=False)

            ax_data.set_title('Input Matrix')

            sns.heatmap(data, ax=ax_data, cmap=cmap,
                                square=True, cbar=False,
                                vmin=0., vmax=1.)

            # Preds

            shape_original = pred.shape

            pred = np.atleast_2d(pred.reshape(-1)).T
            pred = MinMaxScaler().fit_transform(pred).T
            pred = pred.reshape(*shape_original)

            matrix_dim = int(math.sqrt(pred.shape[0]))

            pred = pred.reshape(matrix_dim, matrix_dim)

            ax_preds.tick_params(labelbottom=False, labelleft=False)

            ax_preds.set_title('Predicted Matrix')

            sns.heatmap(pred, ax=ax_preds, cmap=cmap,
                                square=True, cbar=False,
                                vmin=0., vmax=1.)

            # Absdiff

            shape_original = diff.shape

            diff = np.atleast_2d(diff.reshape(-1)).T
            diff = data_scaler.transform(diff).T
            diff = diff.reshape(*shape_original)

            matrix_dim = int(math.sqrt(diff.shape[0]))

            diff = diff.reshape(matrix_dim, matrix_dim)

            ax_diff.tick_params(labelbottom=False, labelleft=False)

            ax_diff.set_title('Normalized Diff')

            sns.heatmap(diff, ax=ax_diff, cmap=cmap,
                                square=True, cbar=False,
                                vmin=0., vmax=1.)
                    
            plt.tight_layout()
                    
        pbar.update(1)

    frames_to_render = np.arange(len(data))

    ani = animation.FuncAnimation(fig, animate,
                                    frames_to_render,
                                    interval=interval)

    Writer = animation.writers[writer_type]

    writer = Writer(fps=fps,
                        metadata=dict(artist=artist_name),
                        codec=codec,
                        extra_args=writer_args)

    ani.save(output_filename, writer)


def render_time_series(output_filename: str,
                            artist_name: str,
                            bucket_count: int = 8,
                            fps: int = 60,
                            interval: int = 1,
                            channels_lower: int = 0,
                            channels_upper: int = -1,
                            fixed_y_lower = None,
                            fixed_y_upper = None,
                            fig_size: tuple = (19.2, 5.4),
                            label_size: float = 14,
                            title_size: float = 20,
                            cmap = 'viridis',
                            fig_facecolor = 'darkgrey',
                            ax_facecolor = 'lightgrey',
                            seaborn_plotting_context: str = 'talk',
                            writer_args: list = ['-qscale:v', '3']) -> None:

    
    if not len(self._data_list):
        raise Warning('No data pushed to AttentionVisualizer.'
                                                'No output generated.')

    sns.set(rc={'figure.figsize': fig_size,
                'figure.titlesize': title_size,
                    'axes.labelsize': label_size,
                    'axes.titlesize': title_size,
                    'figure.facecolor': fig_facecolor,
                    'axes.facecolor': ax_facecolor})

    pbar = tqdm(total=len(self._attentions_list),
                    desc='Rendering attention projection')

    fig = plt.figure()


    def animate(position):

        start = position
        end = position + self._attentions_list[position][2].shape[1]

        window = self._data_list[position].squeeze()

        window = window[:, channels_lower:channels_upper]

        attn = self._attentions_list[position][2][0, :, channels_lower:channels_upper]

        shape_original = attn.shape

        attn = np.atleast_2d(attn.reshape(-1)).T
        attn = MinMaxScaler().fit_transform(attn).T
        attn = attn.reshape(*shape_original)

        plt.clf()

        ax = fig.subplots()

        with sns.plotting_context(seaborn_plotting_context):

            x = np.arange(start, end)

            ax.plot(x, window)

            ax.set_title('Input Time Series')

            ax.set_xlabel('')
            ax.set_ylabel('')

            pbar.update(1)

    frames_to_render = np.arange(len(self._attentions_list))

    ani = animation.FuncAnimation(fig, animate,
                                    frames_to_render,
                                    interval=interval)

    Writer = animation.writers[writer_type]
    
    writer = Writer(fps=fps, metadata=dict(artist=artist_name),
                            codec=codec, extra_args=writer_args)

    ani.save(output_filename, writer)


if __name__ == '__main__':

    for epoch in range(4):
        data = load_numpy_array(f'data/data_train_mscred_epoch_{epoch}.npy')
        pred = load_numpy_array(f'data/preds_train_mscred_epoch_{epoch}.npy')

        render_autocorr_data_pred(data, pred, f'animations/train_{epoch}.mp4', '')