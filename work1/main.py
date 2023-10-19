import argparse
from util import get_dataset,split_dataset,build
parser = argparse.ArgumentParser(
                    prog='P&NPC',
                    description='Parametric & Non-Parametric Classifiers')
parser.add_argument('-m','--model',default='LDA')
parser.add_argument('-d','--dataset',default='heart_disease')


if __name__ == '__main__':
    args = parser.parse_args()
    print(args.model, args.dataset)
    dataset = get_dataset(args.dataset)
    X_train, X_test, y_train, y_test  = split_dataset()
    model = build(args.model)
    model.fit(X_train,y_train)
    model.predict(X_test,y_test)
    