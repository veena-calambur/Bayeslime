import bayeslime
import bayeslime.lime_tabular
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np

np.random.seed(0)

def tabular_example():

	# Loading data and fitting model
	iris = sklearn.datasets.load_iris()
	train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(iris.data, iris.target, train_size=0.80)
	rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
	rf.fit(train, labels_train)

	# Generating a bayeslime explanation
	explainer = bayeslime.lime_tabular.LimeTabularExplainer(train,
													   feature_names=iris.feature_names, 
													   class_names=iris.target_names, 
													   discretize_continuous=True)

	# Generating the bayes lime explanation
	# sampling for a fixed number of times
	exp = explainer.explain_instance(test[1], 
		 						     rf.predict_proba, 
		 						     num_features=4, 
		 						     labels=(1,),
		 						     num_samples=1_000,
		 						     percent=95)

	# Gene
	fig = exp.as_pyplot_figure(label=1)
	plt.tight_layout()
	plt.savefig("sample_with_fixed_amount.png")
	plt.cla()

	string = "Sampling with a predefined sampling amount we get\n"
	string += "the following feature importances and 95% credible intervals:"
	print (string)
	print (exp.as_list())

	# Generating an explanation sampling until we have found a confident
	# explanation.  This means we sample until we are 95% sure we have 
	# found the correct feature importance ranking.
	exp = explainer.explain_instance(test[1], 
		 						     rf.predict_proba, 
		 						     sample_until_confident=True,
		 						     num_features=4, 
		 						     labels=(1,),
		 						     num_samples=1_000,
		 						     percent=95)

	fig = exp.as_pyplot_figure(label=1)
	plt.tight_layout()
	plt.savefig("sample_until_confident.png")

	string = "Sampling until we are 95% confident we have found the correct feature importances\n"
	string += "the following feature importances and 95% credible intervals:"
	print (string)
	print (exp.as_list())


tabular_example()