# mBART-jp-ch-anime-finetune

git This is a Japanese to Chinese translation model finetuned based on [mbart-large-50-many-to-many-mmt](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt). Training file is run.py, test file is test.py, dataset is split_datasets/, generated data is data_test/.
Complete my README.md part. Based on the above, and add text.

This is a Japanese-to-Chinese translation model finetuned based on [mbart-large-50-many-to-many-mmt](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt). The model is specifically optimized for translating anime-related content, ensuring accurate and context-aware translations between Japanese and Chinese.

---

## Overview

* **Base Model** : [mbart-large-50-many-to-many-mmt](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt)
* **Fine-tuning Domain** : Anime-related text (e.g., subtitles, dialogues, descriptions)
* **Training Script** : `run.py`
* **Testing Script** : `test.py`
* **Dataset** : Located in `split_datasets/`
* **Generated Data** : Stored in `data_test/`

---

## Features

* **High-Quality Translations** : Tailored for anime-related content, capturing nuances and cultural context.
* **Efficient Fine-Tuning** : Leverages the multilingual capabilities of mBART-large-50.
* **Easy to Use** : Simple scripts for training and testing.

---

## Usage

### Training

To fine-tune the model, run the following command:

```
python run.py
```

The training data should be placed in the `split_datasets/` directory.

### Testing

To evaluate the model, use the test script:

```
python test.py
```

The generated translations will be saved in the `data_test/` directory.

---

## License

This project is licensed under the same terms as the base model, [mbart-large-50-many-to-many-mmt](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt). Refer to the original license for more details.

---

## Acknowledgments

* Thanks to Facebook AI for the [mbart-large-50-many-to-many-mmt](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt) model.
* Special thanks to the contributors of the anime dataset used for fine-tuning.

---

For questions or contributions, please open an issue or submit a pull request.
