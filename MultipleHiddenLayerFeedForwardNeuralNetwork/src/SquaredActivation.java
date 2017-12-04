
public class SquaredActivation implements ActivationFunction{

	@Override
	public double derivative(double net) {
		if(net >= 0) {
			return 2*net;
		}else {
			return -2*net;
		}
	}

	@Override
	public double function(double net) {
		if(net >= 0) {
			return net*net;
		}else {
			return -1*net*net;
		}
	}

}
