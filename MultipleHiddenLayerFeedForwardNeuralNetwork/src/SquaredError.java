
public class SquaredError implements ErrorFunction{

	@Override
	public double derivative(double net, double target, ActivationFunction activationFunction) {
		return -1*(target-activationFunction.function(net))*activationFunction.derivative(net);
	}

	@Override
	public double function(double output, double target) {
		double diff = target-output;
		return 0.5*diff*diff;
	}

}
