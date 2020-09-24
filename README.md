# Docker-TensorFlowTTS-zh
基于TensorFlowTTS的中文CPU实时TTS Demo，在CPU服务器部署后可通过网页交互体验

Docker image for [TensorFlowTTS-zh-using-baker-dataset](https://colab.research.google.com/drive/1YpSHRBRPBI7cnTkQn1UcVTWEQVbsUm1S) from [TensorFlowTTS](https://github.com/TensorSpeech/TensorFlowTTS)

使用 [@azraelkuan](https://github.com/azraelkuan) 提供的中文预训练模型（tacotron2 + MB-MelGAN）

## Using

```sh
$ docker run -it -p 5000:5000 jackiexiao/tensorflowtts:zh_baker

```

Visit http://localhost:5000 for web interface.

Do HTTP GET at http://localhost:5000/api/tts?text=your%20sentence to get WAV audio back:

```sh
$ curl -G --output - \
    --data-urlencode 'text=你好，世界！' \
    'http://localhost:5000/api/tts' | \
    aplay
```

## Building Yourself

The Docker image is built using [these instructions](https://colab.research.google.com/drive/1YpSHRBRPBI7cnTkQn1UcVTWEQVbsUm1S). See Dockerfile for more details.


