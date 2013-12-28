package pls.othertools.niftiextractor;

/**
 * Exception class that is generated when an error occurs when attempting
 * to extract the eigen images out of result files. This exception will be
 * thrown in cases where certain fields do not exist when they should i.e
 * when st_cords should be present. Other cases may exist as well, such as
 * when the file chosen to extract images is not even a result file.
 */
public class NiftiExtractorException extends Exception {
	private String message;
	
	public NiftiExtractorException(String message) {
		this.message = message;
	}
	
	@Override
	public String getMessage(){
		return message;
	}
}
