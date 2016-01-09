import combdetection.utils.trainer as tr
import combdetection.models as md
import sys

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    nn_name = sys.argv[2]

    model = md.get_saliency_network(train=True)
    trainer = tr.Trainer()
    hist = trainer.fit(model, dataset_name,nn_name,nb_epoch=10)
    trainer.show_training_performance(hist)