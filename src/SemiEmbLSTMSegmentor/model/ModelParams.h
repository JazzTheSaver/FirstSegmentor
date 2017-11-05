#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:

	LookupTable chars;
	LookupTable l_bichars;
	LookupTable r_bichars;

	LookupTable chars_r;
	LookupTable l_bichars_r;
	LookupTable r_bichars_r;

	LSTM1Params lstm_left_param;
	LSTM1Params lstm_right_param;
	
	UniParams NonLinear;
	UniParams character;
	UniParams olayer_linear; // output
public:
	Alphabet charsAlpha;
	Alphabet r_charsAlpha;
	Alphabet bicharsAlpha;
	Alphabet r_bicharsAlpha;
	Alphabet labelAlpha; // should be initialized outside
	SoftMaxLoss loss;


public:
	bool initial(HyperParams& opts){

		// some model parameters should be initialized outside
		if (chars.nVSize <= 0 || labelAlpha.size() <= 0){
			return false;
		}
		opts.charDim = chars.nDim;
		opts.bicharDim = l_bichars.nDim;
		opts.labelSize = labelAlpha.size();

		NonLinear.initial(opts.characterSize, opts.charDim + opts.characterSize + opts.bicharDim + opts.characterSize, true);

		lstm_left_param.initial(opts.characterSize , opts.characterSize);
		lstm_right_param.initial(opts.characterSize , opts.characterSize);
		
		character.initial(opts.characterSize,opts.characterSize * 2, true);

		olayer_linear.initial(opts.labelSize, opts.characterSize, false);
		
	//	hidden_linear.initial(opts.hiddenSize, opts.windowOutput, true);
	//	opts.inputSize = opts.hiddenSize ;
		
		return true;
	}

	void exportModelParams(ModelUpdate& ada){
		chars.exportAdaParams(ada);
		l_bichars.exportAdaParams(ada);
		r_bichars.exportAdaParams(ada);
		
		chars_r.exportAdaParams(ada);
		l_bichars_r.exportAdaParams(ada);
		r_bichars_r.exportAdaParams(ada);

		lstm_left_param.exportAdaParams(ada);
		lstm_right_param.exportAdaParams(ada);
		NonLinear.exportAdaParams(ada);
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