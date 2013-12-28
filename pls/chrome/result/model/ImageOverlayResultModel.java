package pls.chrome.result.model;

public class ImageOverlayResultModel extends ResultModel {

	public String getVariableType() {
		return "";
	}
	
	public String getAbbrVariableType() {
		return "";
	}
	
	protected void addToRepository(GeneralRepository repository) {
		repository.addModel(getFilename(), this);
	}
	
	protected void removeFromRepository(GeneralRepository repository) {
		repository.removeImageOverlayModel(getFilename());
	}
	
}
