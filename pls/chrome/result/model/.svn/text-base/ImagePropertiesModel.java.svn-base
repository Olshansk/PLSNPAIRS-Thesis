package pls.chrome.result.model;

import java.awt.Color;

import pls.chrome.result.controller.Publisher;
import pls.chrome.result.controller.observer.brainimageproperties.CrosshairColourEvent;
import pls.chrome.result.controller.observer.brainimageproperties.CrosshairTransparencyEvent;
import pls.chrome.result.controller.observer.brainimageproperties.LabelsColourEvent;
import pls.chrome.result.controller.observer.brainimageproperties.LabelsTransparencyEvent;
import pls.chrome.result.controller.observer.brainimageproperties.UseCrosshairEvent;
import pls.chrome.result.controller.observer.brainimageproperties.UseDescriptionLabelsEvent;
import pls.chrome.result.controller.observer.brainimageproperties.UseLabelsEvent;
import pls.chrome.result.controller.observer.singlebrainimageview.BrainViewEvent;
import pls.chrome.result.controller.observer.singlebrainimageview.RotationEvent;
import pls.chrome.result.controller.observer.singlebrainimageview.ZoomEvent;

public class ImagePropertiesModel {
	int mBrainView = BrainData.AXIAL;
	
	Color mCrosshairColour = Color.GREEN;
	boolean mCrosshairEnabled = true;
	int mCrosshairTransparency = 255;

	Color mLabelsColour = Color.MAGENTA;
	boolean mLabelsEnabled = true;
	int mLabelsTransparency = 255;
	
	boolean mDescriptionLabelsEnabled = true;
	
	int mRotation = 0;
	
	double mZoomLevel = 1.0;

	private Publisher mPublisher;
	
	public ImagePropertiesModel(Publisher publisher) {
		mPublisher = publisher;
	}
	
	public void setBrainView(int brainView) {
		mBrainView = brainView;
		
		mPublisher.publishEvent(new BrainViewEvent() );
	}
	
	public int getBrainView() {
		return mBrainView;
	}
	
	public boolean isCrosshairEnabled() {
		return mCrosshairEnabled;
	}
	
	public void setCrosshairEnabled(boolean enabled) {
		mCrosshairEnabled = enabled;
		
		mPublisher.publishEvent(new UseCrosshairEvent() );
	}
	
	public int getCrosshairTransparency() {
		return mCrosshairTransparency;
	}

	public void setCrosshairTransparency(int crosshairTransparency) {
		mCrosshairTransparency = crosshairTransparency;
		
		mPublisher.publishEvent(new CrosshairTransparencyEvent() );
	}
	
	public Color getCrosshairColour() {
		return mCrosshairColour;
	}

	public void setCrosshairColour(Color crosshairColour) {
		mCrosshairColour = crosshairColour;
		
		mPublisher.publishEvent(new CrosshairColourEvent() );
	}

	public boolean labelsEnabled() {
		return mLabelsEnabled;
	}
	
	public void setLabelsEnabled(boolean enabled) {
		mLabelsEnabled = enabled;
		
		mPublisher.publishEvent(new UseLabelsEvent() );
	}

	public Color getLabelsColour() {
		return mLabelsColour;
	}

	public void setLabelsColour(Color colour) {
		mLabelsColour = colour;
		
		mPublisher.publishEvent(new LabelsColourEvent() );
	}

	public void setLabelsTransparency(int transparency) {
		mLabelsTransparency = transparency;
		
		mPublisher.publishEvent(new LabelsTransparencyEvent() );
	}

	public int getLabelsTransparency() {
		return mLabelsTransparency;
	}
	
	public boolean descriptionLabelsEnabled() {
		return mDescriptionLabelsEnabled;
	}
	
	public void setDescriptionLabelsEnabled(boolean enabled) {
		mDescriptionLabelsEnabled = enabled;
		
		mPublisher.publishEvent(new UseDescriptionLabelsEvent() );
	}
	
	public int getRotation() {
		return mRotation;
	}
	
	public void setRotation(int rotation) {
		mRotation = rotation;
		
		mRotation %= 4;
		
		if (mRotation < 0) {
			mRotation += 4;
		}
		
		mPublisher.publishEvent(new RotationEvent() );
	}

	public double getZoom() {
		return mZoomLevel;
	}

	public void setZoom(double zoom) {
		mZoomLevel = zoom;
		
		mPublisher.publishEvent(new ZoomEvent() );
	}
}
