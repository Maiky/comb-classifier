import combdetection.config as conf
if not conf.ANALYSE_PLOTS_SHOW:
    #needed to plot images on flip
    import matplotlib
    matplotlib.use('Agg')
import combdetection.utils.trainer as tr
import combdetection.models as md
import sys


if __name__ == '__main__':
    dataset_name = sys.argv[1]
    nn_name = sys.argv[2]

    model = md.get_saliency_network(train=True)
    trainer = tr.Trainer()
    hist, X_test, Y_test = trainer.fit(model, dataset_name,nn_name,nb_epoch=100)
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    trainer.show_training_performance(hist)
