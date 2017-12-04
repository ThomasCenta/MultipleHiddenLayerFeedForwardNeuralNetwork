
public class SoftMaxActivation implements ActivationFunction{

	private double xTranslation;
	private double xElongation;
	
	public SoftMaxActivation(double xTranslation, double xElongation) {
		assert xElongation > 0;
		
		this.xElongation = xElongation;
		this.xTranslation = xTranslation;
	}
	
	@Override
	public double derivative(double net) {
		double input = this.xElongation*net - this.xTranslation;
		return this.xElongation*1.0/(1.0+Math.exp(-1.0*input));
	}

	@Override
	public double function(double net) {
		double input = this.xElongation*net - this.xTranslation;
		return Math.log(1+Math.exp(input));
	}

}
