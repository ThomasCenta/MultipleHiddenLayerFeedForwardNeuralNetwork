
public class SigmoidActivation implements ActivationFunction{

	private double xElongation;
	private double xTranslation;
	
	
	public SigmoidActivation(double xElongation, double xTranslation) {
		this.xElongation = xElongation;
		this.xTranslation = xTranslation;
	}
	
	@Override
	public double derivative(double net) {
		double input = this.xElongation*net - this.xTranslation;
		return this.xElongation*this.function(input)*(1.0-this.function(input));
	}

	@Override
	public double function(double net) {
		double input = this.xElongation*net-this.xTranslation;
		return 1.0/(1+Math.exp(-1*input));
	}

}
