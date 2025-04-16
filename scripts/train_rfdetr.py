

from rfdetr import RFDETRBase

model = RFDETRBase(resolution=560,num_classes=21,pretrain_weights = None) # resolution shoud be divisible by 56
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    model.train(dataset_dir="data/100k", epochs=10, batch_size=1, grad_accum_steps=4, lr=1e-4, output_dir="experiments/v1",)