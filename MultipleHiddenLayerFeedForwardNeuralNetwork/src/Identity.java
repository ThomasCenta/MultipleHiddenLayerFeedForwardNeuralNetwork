
public class Identity implements ActivationFunction{

	@Override
	public double derivative(double net) {
		return 1.0;
	}

	@Override
	public double function(double net) {
		return net;
	}

}
