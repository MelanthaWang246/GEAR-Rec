# GEAR-Rec
GEAR-Rec官方仓库，内含原数据集和训练/测试数据。同时提供包括数据处理、模型训练等在内的代码。

## data
储存经原始数据集初步处理后的数据和各阶段输出，和处理代码。由于上传限制的问题，剩余文件需要从https://drive.google.com/drive/folders/1P-pYA9LoUJsrQyzfgVwcaNdh4ZbHWw_t?usp=sharing下载。
## data_preprocess
构建大模型输入
## llm_code
构建用户和物品的语义描述
## rec_code
训练和测试代码。启动时需要cd main_code再train_mooc.sh或者train_ml-1m.sh
