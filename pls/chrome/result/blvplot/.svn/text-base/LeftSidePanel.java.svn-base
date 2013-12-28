package pls.chrome.result.blvplot;

import pls.chrome.CollapsibleList;
import pls.chrome.result.controller.observer.Event;
import pls.chrome.result.controller.observer.brainimageproperties.BackgroundImageEvent;
import pls.chrome.result.controller.observer.brainimageproperties.BrainImagePropertiesObserver;
import pls.chrome.result.controller.observer.brainimageproperties.CrosshairColourEvent;
import pls.chrome.result.controller.observer.brainimageproperties.CrosshairTransparencyEvent;
import pls.chrome.result.controller.observer.brainimageproperties.LabelsColourEvent;
import pls.chrome.result.controller.observer.brainimageproperties.LabelsTransparencyEvent;
import pls.chrome.result.controller.observer.brainimageproperties.UseCrosshairEvent;
import pls.chrome.result.controller.observer.brainimageproperties.UseDescriptionLabelsEvent;
import pls.chrome.result.controller.observer.brainimageproperties.UseLabelsEvent;
import pls.chrome.result.controller.observer.colourscale.ColourScaleEvent;
import pls.chrome.result.controller.observer.colourscale.ColourScaleObserver;
import pls.chrome.result.controller.observer.datachange.DataChangeObserver;
import pls.chrome.result.controller.observer.datachange.FlipVolumeEvent;
import pls.chrome.result.controller.observer.datachange.InvertedLvEvent;
import pls.chrome.result.controller.observer.datachange.LoadedVolumesEvent;
import pls.chrome.result.controller.observer.filters.BrainFilterEvent;
import pls.chrome.result.controller.observer.filters.FiltersObserver;
import pls.chrome.result.controller.observer.filters.IncorrectLagsSelectedEvent;
import pls.chrome.result.controller.observer.filters.SliceFiltersEvent;
import pls.chrome.result.controller.observer.filters.ViewedLvsEvent;
import pls.chrome.result.controller.observer.selection.SelectedDataTypeChangedEvent;
import pls.chrome.result.controller.observer.selection.SelectedLvChangedEvent;
import pls.chrome.result.controller.observer.selection.SelectedVolumeChangedEvent;
import pls.chrome.result.controller.observer.selection.SelectionEvent;
import pls.chrome.result.controller.observer.selection.SelectionObserver;
import pls.chrome.result.controller.observer.singlebrainimageview.BrainViewEvent;
import pls.chrome.result.controller.observer.singlebrainimageview.RotationEvent;
import pls.chrome.result.controller.observer.singlebrainimageview.SingleBrainImageViewObserver;
import pls.chrome.result.controller.observer.singlebrainimageview.ZoomEvent;
import pls.chrome.result.model.GeneralRepository;

@SuppressWarnings("serial")
public class LeftSidePanel extends CollapsibleList implements BrainImagePropertiesObserver,
ColourScaleObserver, DataChangeObserver, FiltersObserver, SelectionObserver,
SingleBrainImageViewObserver {
	VolumeBrowser mVolumeBrowser = null;
	FilterBrowser mFilterBrowser = null;
	ColourScalePanel mColourScalePanel = null;
	CoordinatesPanel mCoordinatesPanel = null;
	ControlPanel mControlPanel = null;
	
	public LeftSidePanel(GeneralRepository repository) {
		repository.getPublisher().registerObserver(this);
		
		mVolumeBrowser = new VolumeBrowser(repository);
		mColourScalePanel = new ColourScalePanel(repository);
		mCoordinatesPanel = new CoordinatesPanel(repository);
		mControlPanel = new ControlPanel(repository);
		mFilterBrowser = new FilterBrowser(repository);
		
		addItem("Volume Browser", mVolumeBrowser);
		addItem("Colour Scale", mColourScalePanel);
		addItem("Voxel Selection", mCoordinatesPanel);
		addItem("Control Panel", mControlPanel);
		addItem("Mask Browser", mFilterBrowser);
		
		mFilterBrowser.refresh();
	}
	
	public void updateSelection() {
		mCoordinatesPanel.updateSelection();
	}
	
	public void updateColourScale() {
		mColourScalePanel.updateColourScale();
	}

	public void refreshPanel() {
//		mDataTypePanel.refreshTree();
		mColourScalePanel.refreshWidgets();
	}

	public void selectedVolumeChanged() {
		updateSelection();
//		mDataTypePanel.selectResultFile();
	}
	
	public void selectedDataTypeChanged() {
//		mDataTypePanel.selectDataType();
//		mMaxMinThreshPanel.setBrainData(dataName, lvNumber);
	}

	public void selectedLvChanged() {
		mVolumeBrowser.selectLv();
		mColourScalePanel.updateColourScale();
	}
	
	public void lagNumbersChanged() {
		mCoordinatesPanel.updateLagComboBox();
		mCoordinatesPanel.updateSelection();
	}

	public void loadVolumes() {
		mVolumeBrowser.refreshTree();
		mCoordinatesPanel.volumesAdded();
		mControlPanel.volumesAdded();
	}
	
	public void updateBrainView() {
		mControlPanel.updateBrainView();
		mCoordinatesPanel.updateBrainView();
	}
	
	private void updateInvertedLv() {
		mVolumeBrowser.updateInvertedLv();
	}
	
	///////////////////////////////////////////////////////////////////////////
	// Event handling
	///////////////////////////////////////////////////////////////////////////
	@Override
	public void notify(Event e) {}

	///////////////////////////////////////////////////////////////////////////
	// Brain image property event handlers
	///////////////////////////////////////////////////////////////////////////
	@Override
	public void notify(BackgroundImageEvent e) {}

	@Override
	public void notify(CrosshairColourEvent e) {}

	@Override
	public void notify(CrosshairTransparencyEvent e) {}

	@Override
	public void notify(LabelsColourEvent e) {}

	@Override
	public void notify(LabelsTransparencyEvent e) {}

	@Override
	public void notify(UseCrosshairEvent e) {}

	@Override
	public void notify(UseDescriptionLabelsEvent e) {}

	@Override
	public void notify(UseLabelsEvent e) {}

	///////////////////////////////////////////////////////////////////////////
	// Colour scale event handlers
	///////////////////////////////////////////////////////////////////////////
	@Override
	public void notify(ColourScaleEvent e) {
		updateColourScale();
	}

	///////////////////////////////////////////////////////////////////////////
	// Selection event handlers
	///////////////////////////////////////////////////////////////////////////
	@Override
	public void notify(SelectedDataTypeChangedEvent e) {
		selectedDataTypeChanged();
	}

	@Override
	public void notify(SelectedLvChangedEvent e) {
		selectedLvChanged();
	}

	@Override
	public void notify(SelectedVolumeChangedEvent e) {
		selectedVolumeChanged();
	}

	@Override
	public void notify(SelectionEvent e) {
		updateSelection();
	}

	///////////////////////////////////////////////////////////////////////////
	// Single brain image viewer event handlers
	///////////////////////////////////////////////////////////////////////////
	@Override
	public void notify(BrainViewEvent e) {
		updateBrainView();
	}

	@Override
	public void notify(RotationEvent e) {}

	@Override
	public void notify(ZoomEvent e) {}

	///////////////////////////////////////////////////////////////////////////
	// Data change event handlers
	///////////////////////////////////////////////////////////////////////////
	@Override
	public void notify(FlipVolumeEvent e) {}

	@Override
	public void notify(LoadedVolumesEvent e) {
		loadVolumes();
	}
	
	@Override
	public void notify(InvertedLvEvent e) {
		updateInvertedLv();
	}

	///////////////////////////////////////////////////////////////////////////
	// Filters event handlers
	///////////////////////////////////////////////////////////////////////////
	@Override
	public void notify(SliceFiltersEvent e) {
		lagNumbersChanged();
	}

	@Override
	public void notify(ViewedLvsEvent e) {}

	@Override
	public void notify(BrainFilterEvent e) {
		mFilterBrowser.refresh();
	}

	public void notify(IncorrectLagsSelectedEvent e){};
}
