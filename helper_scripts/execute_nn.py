import sys
import combdetection.utils.generator as generator
import combdetection.neuralnet
import combdetection.config
import numpy as np
import pickle
import os.path



if __name__ == '__main__':
    dataset_file = sys.argv[1]
    config_file = ""
    if(len(sys.argv) >=3):
        config_file = sys.argv[2]


    gen = generator.Generator(dataset_file)
    X_train, X_test, y_train, y_test= gen.load_traindata()
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    y_train = np.asarray(y_train)
    print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
    print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

    if (len(config_file) == 0) | (not os.path.exists(config_file)):
        nn = combdetection.neuralnet.NeuralNetMLP(n_output=10,
                          n_features=X_train.shape[1],
                          n_hidden=50,
                          l2=0.1,
                          l1=0.0,
                          epochs=1000,
                          eta=0.001,
                          alpha=0.001,
                          decrease_const=0.00001,
                          minibatches=50,
                          random_state=1)
        nn.fit(X_train, y_train, print_progress=True)
        pickle.dump(nn, open(config_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    else:
        nn = pickle.load(open(config_file, "rb"))

    import matplotlib.pyplot as plt
    plt.plot(range(len(nn.cost_)), nn.cost_)
    plt.ylim([0, 2000])
    plt.ylabel('Cost')
    plt.xlabel('Epochs * 50')
    plt.tight_layout()
    # plt.savefig('./figures/cost.png', dpi=300)
    plt.show()

    batches = np.array_split(range(len(nn.cost_)), 1000)
    cost_ary = np.array(nn.cost_)
    cost_avgs = [np.mean(cost_ary[i]) for i in batches]

    plt.plot(range(len(cost_avgs)), cost_avgs, color='red')
    plt.ylim([0, 2000])
    plt.ylabel('Cost')
    plt.xlabel('Epochs')
    #plt.tight_layout()
    #plt.savefig('./figures/cost2.png', dpi=300)
    plt.show()

    y_train_pred = nn.predict(X_train)
    acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
    print('Training accuracy: %.2f%%' % (acc * 100))

    y_test_pred = nn.predict(X_test)
    acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
    print('Training accuracy: %.2f%%' % (acc * 100))

    miscl_img = X_test[y_test != y_test_pred][:25]
    correct_lab = y_test[y_test != y_test_pred][:25]
    miscl_lab= y_test_pred[y_test != y_test_pred][:25]

    labels = combdetection.config.NETWORK_CLASS_LABELS

    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
    ax = ax.flatten()
    for i in range(25):
        img = miscl_img[i].reshape(combdetection.config.GENERATOR_SAMPLE_SIZE[0], combdetection.config.GENERATOR_SAMPLE_SIZE[1])
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        right_l =[label for label, enc in labels.items() if enc == correct_lab[i]]
        pre_l = [label for label, enc in labels.items() if enc == miscl_lab[i]]
        ax[i].set_title('%d) t: %s p: %s' % (i+1, right_l[0] ,pre_l[0]))

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    # plt.savefig('./figures/mnist_miscl.png', dpi=300)
    plt.show()
