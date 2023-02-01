# NaturalSpeech: End-to-End Text to Speech Synthesis with Human-Level Quality

This is an implementation of Microsoft's [NaturalSpeech: End-to-End Text to Speech Synthesis with Human-Level Quality](https://arxiv.org/abs/2205.04421) in Pytorch.

Contribution and pull requests are highly appreciated!

23.02.01: Pretrained models or demo samples will soon be released.


### Overview

![figure1](resources/figure1.png)

Naturalspeech is a VAE-based model that employs several techniques to improve the prior and simplify the posterior. It differs from VITS in several ways, including:
- **Phoneme pre-training**: Naturalspeech uses a pre-trained phoneme encoder on a large text corpus, obtained through masked language modeling on phoneme sequences.
- **Differentiable durator**: The posterior operates at the frame level, while the prior operates at the phoneme level. Naturalspeech uses a differentiable durator to bridge the length difference, resulting in soft and flexible features that are expanded.
- **Bidirectional Prior/Posterior**: Naturalspeech reduces the posterior and enhances the prior through normalizing flow, which maps in both directions with forward and backward loss.
- **Memory-based VAE**: The prior is further enhanced through a memory bank using Q-K-V attention."


### Notes
- This implementation does not include pre-training of phonemes using a large-scale text corpus from the news-crawl dataset.
- The multiplier for each loss can be adjusted in the configuration file. Using losses without a multiplier may not lead to convergence.
- The tuning stage for the last 2k epochs has been omitted.
- Due to the high VRAM usage of the soft-dtw loss, there is an option to use a non-softdtw loss for memory efficiency.
- For the soft-dtw loss, the warp factor has been set to 134.4 (0.07 * 192) to match the non-softdtw loss, instead of 0.07.
- To train the duration predictor in the warm-up stage, duration labels are required. The paper suggests using any tool to provide the duration label. In this implementation, a pre-trained VITS model was used.
- To further improve memory efficiency during training, randomly silced sequences are fed to the decoder as in the VITS model.




### How to train

0.
    ```
    # python >= 3.6
    pip install -r requirements.txt
    ```

1. clone this repository
1. download **The LJ Speech Dataset**: [link](https://keithito.com/LJ-Speech-Dataset/)
1. create symbolic link to ljspeech dataset: 
    ```
    ln -s /path/to/LJSpeech-1.1/wavs/ DUMMY1
    ```
1. text preprocessing (optional, if you are using custom dataset):
    1. `apt-get install espeak`
    2. 
        ```
        python preprocess.py --text_index 1 --filelists filelists/ljs_audio_text_train_filelist.txt filelists/ljs_audio_text_val_filelist.txt filelists/ljs_audio_text_test_filelist.txt
        ```

1. duration preprocessing (obtain duration labels using pretrained VITS):
    > If you want to skip this section, use `durations/durations.tar.bz2` and overwrite the `durations` folder.
    1. `git clone https://github.com/jaywalnut310/vits.git; cd vits`
    2. create symbolic link to ljspeech dataset 
        ```
        ln -s /path/to/LJSpeech-1.1/wavs/ DUMMY1
        ```
    3. download pretrained VITS model described as from VITS official github: [github link](https://github.com/jaywalnut310/vits) / [pretrained models](https://drive.google.com/drive/folders/1ksarh-cJf3F5eKJjLVWY0X1j1qsQqiS2)
    4. setup monotonic alignment search (for VITS inference): 
        ```
        cd monotonic_align; mkdir monotonic_align; python setup.py build_ext --inplace; cd ..
        ```
    5. copy duration preprocessing script to VITS repo: `cp /path/to/naturalspeech/preprocess_durations.py .`
    6. 
        ```
        python3 preprocess_durations.py --weights_path ./pretrained_ljs.pth --filelists filelists/ljs_audio_text_train_filelist.txt.cleaned filelists/ljs_audio_text_val_filelist.txt.cleaned filelists/ljs_audio_text_test_filelist.txt.cleaned
        ```
    7. once the duration labels are created, copy the labels to the naturalspeech repo: `cp -r durations/ path/to/naturalspeech`

1. train (warmup)
    ```
    python3 train.py -c configs/ljs.json -m [run_name] --warmup
    ```
    Note here that `ljs.json` is for low-resource training, which runs for 1500 epochs and does not use soft-dtw loss. If you want to reproduce the steps stated in the paper, use `ljs_reproduce.json`, which runs for 15000 epochs and uses soft-dtw loss.

1. initialize and attach memory bank after warmup:
    ```
      python3 attach_memory_bank.py -c configs/ljs.json --weights_path logs/[run_name]/G_xxx.pth
    ```
    if you lack memory, you can specify the `--num_samples` argument to use only a subset of samples.

1. train (resume)
    ```
      python3 train.py -c configs/ljs.json -m [run_name]
    ```

You can use tensorboard to monitor the training.
```
tensorboard --logdir /path/to/naturalspeech/logs
```

During each evaluation phase, a selection of samples from the test set is evaluated and saved in the `logs/[run_name]/eval` directory.



## References
- [VITS implemetation](https://github.com/jaywalnut310/vits) by @jaywalnut310 for normalizing flows, phoneme encoder, and hifi-gan decoder implementation
- [Parallel Tacotron 2 Implementation](https://github.com/keonlee9420/Parallel-Tacotron2) by @keonlee9420 for learnable upsampling Layer
- [soft-dtw implementation](https://github.com/Maghoumi/pytorch-softdtw-cuda) by @Maghoumi for sdtw loss
