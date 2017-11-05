#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:

	LookupTable chars;

	LSTM1Params lstm_param;
	UniParams character;
	UniParams olayer_linear; // output
public:
	Alphabet charsAlpha;
	Alphabet labelAlpha; // should be initialized outside
	SoftMaxLoss loss;


public:
	bool initial(HyperParams& opts){

		// some model parameters should be initialized outside
		if (chars.nVSize <= 0 || labelAlpha.size() <= 0){
			return false;
		}
		opts.charDim = chars.nDim;
		opts.labelSize = labelAlpha.size();
		
		lstm_param.initial(opts.characterSize, opts.charDim);

		character.initial(opts.characterSize,opts.characterSize, true);

		olayer_linear.initial(opts.labelSize, opts.characterSize, false);
		
	//	hidden_linear.initial(opts.hiddenSize, opts.windowOutput, true);
	//	opts.inputSize = opts.hiddenSize ;
		
		return true;
	}

	void exportModelParams(ModelUpdate& ada){
		chars.exportAdaParams(ada);
		lstm_param.exportAdaParams(ada);
		character.exportAdaParams(ada);
		olayer_linear.exportAdaParams(ada);
	}


	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&chars.E, "words E");
		checkgrad.add(&character.W, "hidden W");
		checkgrad.add(&character.b, "hidden b");
		checkgrad.add(&olayer_linear.W, "output layer W");
	}

	// will add it later
	void saveModel(std::ofstream &os) const{
	/*	wordAlpha.write(os);
		words.save(os);
		hidden_linear.save(os);
		olayer_linear.save(os);
		labelAlpha.write(os);*/
	}

	void loadModel(std::ifstream &is){
		//wordAlpha.read(is);
		//words.load(is, &wordAlpha);
		//hidden_linear.load(is);
		//olayer_linear.load(is);
		//labelAlpha.read(is);
	}

};

#endif /* SRC_ModelParams_H_ */