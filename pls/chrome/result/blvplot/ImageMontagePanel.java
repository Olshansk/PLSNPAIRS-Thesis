package pls.chrome.result.blvplot;
import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.event.MouseEvent;
import java.awt.geom.Point2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Set;

import javax.swing.BoxLayout;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.border.LineBorder;
import javax.swing.event.MouseInputListener;

import pls.chrome.result.controller.ResultsCommandManager;
import pls.chrome.result.model.BrainData;
import pls.chrome.result.model.ColourScaleModel;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.ResultModel;
import pls.chrome.result.model.ViewModel;
import pls.shared.NpairsfMRIResultFileFilter;

@SuppressWarnings("serial")
public class ImageMontagePanel extends JPanel implements MouseInputListener {

	private double scale = 1.0;
	private int rotation = 0;

	private boolean useCrosshair = true;
	private boolean useLabel = true;
	private boolean showOtherLabels = true;
	
	// [volume][datatype][lv#][lag#][slice#]
	public HashMap<String,
			HashMap<String,
              HashMap<Integer,
			    HashMap<Integer,
				  HashMap<Integer, BrainImage>>>>> mBrainImages
				    = new HashMap<String,
					        HashMap<String,
							 HashMap<Integer,
							  HashMap<Integer,
							   HashMap<Integer, BrainImage>>>>>();
	
	protected GeneralRepository mRepository = null;
	// The default transparency value of the crosshair is 255 (opaque)
	private int crosshairTransparency = 255;
	
	// Green is the default crosshair color.
	private Color crosshairColor = Color.GREEN;
	
	// The default transparency value of the labels is 255 (opaque)
	private int labelTransparency = 255;
	
	// Magenta is the default label color.
	private Color labelColor = Color.MAGENTA;
	
	public BrainImage currentlySelected = null;

	ColorGradient mColourGradient;
	public NumbersPanel numbersPanel;
	
	ImageMontagePanel(GeneralRepository repository, ColorGradient colGrad) {
		mRepository = repository;
		mColourGradient = colGrad;
		
		setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
		
		constructImages();
	}
	
	public void updateSliceFilters() {
		constructImages();
	}
	
	public void reInitializeImages(String resultFile) {
		
		HashMap<String, HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>>> fileMap = mBrainImages.get(resultFile);
		for (String dataType : fileMap.keySet() ) {
			HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>> typeMap = fileMap.get(dataType);
			for (Integer lv : typeMap.keySet() ) {
				HashMap<Integer, HashMap<Integer, BrainImage>> lvMap = typeMap.get(lv);
				for (Integer lag : lvMap.keySet() ) {
					HashMap<Integer, BrainImage> lagMap = lvMap.get(lag);
					for (BrainImage slice : lagMap.values() ) {
						slice.reInitializeImage();
					}
				}
			}
		}
	}
	
	public void setCrosshair(boolean useCrosshair) {
		this.useCrosshair = useCrosshair;
		
		for (String resultFile : mBrainImages.keySet() ) {
			HashMap<String, HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>>> fileMap = mBrainImages.get(resultFile);
			for (String dataType : fileMap.keySet() ) {
				HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>> typeMap = fileMap.get(dataType);
				for (Integer lv : typeMap.keySet() ) {
					HashMap<Integer, HashMap<Integer, BrainImage>> lvMap = typeMap.get(lv);
					for (Integer lag : lvMap.keySet() ) {
						HashMap<Integer, BrainImage> lagMap = lvMap.get(lag);
						for (BrainImage slice : lagMap.values() ) {
							slice.useCrosshair = useCrosshair;
						}
					}
				}
			}
		}
		
		if (currentlySelected != null) {
//			currentlySelected.reInitializeImage();
			currentlySelected.repaint();
		}
	}
	
	public void setCrosshairTransparency(int value) {
		crosshairTransparency = value;
		
		for (String resultFile : mBrainImages.keySet() ) {
			HashMap<String, HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>>> fileMap = mBrainImages.get(resultFile);
			for (String dataType : fileMap.keySet() ) {
				HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>> typeMap = fileMap.get(dataType);
				for (Integer lv : typeMap.keySet() ) {
					HashMap<Integer, HashMap<Integer, BrainImage>> lvMap = typeMap.get(lv);
					for (Integer lag : lvMap.keySet() ) {
						HashMap<Integer, BrainImage> lagMap = lvMap.get(lag);
						for (BrainImage slice : lagMap.values() ) {
							slice.crosshairTransparency = value;
						}
					}
				}
			}
		}
		
		if (currentlySelected != null) {
//			currentlySelected.reInitializeImage();
			currentlySelected.repaint();
		}
	}
	
	public void setCrosshairColor(Color color) {
		crosshairColor = color;

		for (String resultFile : mBrainImages.keySet() ) {
			HashMap<String, HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>>> fileMap = mBrainImages.get(resultFile);
			for (String dataType : fileMap.keySet() ) {
				HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>> typeMap = fileMap.get(dataType);
				for (Integer lv : typeMap.keySet() ) {
					HashMap<Integer, HashMap<Integer, BrainImage>> lvMap = typeMap.get(lv);
					for (Integer lag : lvMap.keySet() ) {
						HashMap<Integer, BrainImage> lagMap = lvMap.get(lag);
						for (BrainImage slice : lagMap.values() ) {
							slice.crosshairColor = color;
						}
					}
				}
			}
		}
		
		if (currentlySelected != null) {
//			currentlySelected.reInitializeImage();
			currentlySelected.repaint();
		}
	}
	
	public void setLabel(boolean useLabel) {
		this.useLabel = useLabel;

		for (String resultFile : mBrainImages.keySet() ) {
			HashMap<String, HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>>> fileMap = mBrainImages.get(resultFile);
			for (String dataType : fileMap.keySet() ) {
				HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>> typeMap = fileMap.get(dataType);
				for (Integer lv : typeMap.keySet() ) {
					HashMap<Integer, HashMap<Integer, BrainImage>> lvMap = typeMap.get(lv);
					for (Integer lag : lvMap.keySet() ) {
						HashMap<Integer, BrainImage> lagMap = lvMap.get(lag);
						for (BrainImage slice : lagMap.values() ) {
							slice.useLabel = useLabel;
							slice.repaint();
						}
					}
				}
			}
		}
	}
	
	public void setLabelTransparency(int value) {
		labelTransparency = value;
		
		for (String resultFile : mBrainImages.keySet() ) {
			HashMap<String, HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>>> fileMap = mBrainImages.get(resultFile);
			for (String dataType : fileMap.keySet() ) {
				HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>> typeMap = fileMap.get(dataType);
				for (Integer lv : typeMap.keySet() ) {
					HashMap<Integer, HashMap<Integer, BrainImage>> lvMap = typeMap.get(lv);
					for (Integer lag : lvMap.keySet() ) {
						HashMap<Integer, BrainImage> lagMap = lvMap.get(lag);
						for (BrainImage slice : lagMap.values() ) {
							slice.labelTransparency = value;
							slice.repaint();
						}
					}
				}
			}
		}
	}
	
	public void setOtherLabels(boolean useLabels) {
		showOtherLabels = useLabels;
		
		constructImages();
	}
	
	public void setLabelColor(Color color) {
		labelColor = color;

		for (String resultFile : mBrainImages.keySet() ) {
			HashMap<String, HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>>> fileMap = mBrainImages.get(resultFile);
			for (String dataType : fileMap.keySet() ) {
				HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>> typeMap = fileMap.get(dataType);
				for (Integer lv : typeMap.keySet() ) {
					HashMap<Integer, HashMap<Integer, BrainImage>> lvMap = typeMap.get(lv);
					for (Integer lag : lvMap.keySet() ) {
						HashMap<Integer, BrainImage> lagMap = lvMap.get(lag);
						for (BrainImage slice : lagMap.values() ) {
							slice.labelColor = color;
							slice.repaint();
						}
					}
				}
			}
		}
	}
	
	//number of slices / number of rows per lag = number of slices per row
	private int getNumSlicesPerRow() {
		int brainView = mRepository.getImagePropertiesModel().getBrainView();
		ArrayList<Integer> sliceNums = mRepository.getControlPanelModel().getSliceNumbers(brainView);
		int numRowsPerLag = mRepository.getControlPanelModel().getNumRowsPerLag(brainView);
		return (int)Math.ceil(sliceNums.size() / (double) numRowsPerLag);
	}

	/**
	 * Constructs the brain images montage. This function is also called 
	 * whenever certain settings are changed where the slices/lags/images 
	 * that should be displayed have changed in some way. 
	 * I.e a SliceFilter event occurs, BrainInfoPane intercepts it, 
	 * ImageMontagePanel.updateSliceFilters() is called which calls 
	 * constructImages(). This event occurs when the "PLOT" button is hit in the
	 * control panel.
	 */
	public void constructImages() {
		removeAll();
		updateImageModel();
		drawImagePanels();
	}

	private void updateImageModel(){
		//Represents the type of view we have of the brain. here we have BrainData.AXIAL
		int brainView = mRepository.getImagePropertiesModel().getBrainView();

		double[] globalColourScale = mRepository.getGlobalColourScale();

		Set<String> modelsInRepository = mRepository.getModels();

		ArrayList<Integer> lagNumbers;
		ArrayList<Integer> sliceNumbers = mRepository.getControlPanelModel().getSliceNumbers(brainView);

		// First remove files which are no longer in the repository
		ArrayList<String> filesToRemove = new ArrayList<String>();
		for (String resultFile : mBrainImages.keySet() ) {
			if (!modelsInRepository.contains(resultFile)) {
				filesToRemove.add(resultFile);
			}
		}

		for (String file : filesToRemove) {
			mBrainImages.remove(file);
		}

		// For each result file in the model
		for (String resultFile : modelsInRepository) {
			HashMap<String, HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>>> currFile = null;

			// Obtain the existing entry
			if (mBrainImages.containsKey(resultFile)) {
				currFile = mBrainImages.get(resultFile);
			}
			// Or add a new entry
			else {
				currFile = new HashMap<String, HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>>>();
				mBrainImages.put(resultFile, currFile);
			}

			// Get the model for this result file
			ResultModel model = mRepository.getGeneral(resultFile);

			// Discover the coordinates for that model
			ArrayList<ArrayList<Integer>> coords = model.getBrainData().getSlices(brainView);

			// For each type in that model
			for (String dataType : model.getBrainDataTypes() ) {
				//[datatype][lv#][lag#][slice#]
				HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>> currType = null;

				// Obtain the existing entry
				if (currFile.containsKey(dataType)) {
					currType = currFile.get(dataType);
				}
				// Or add a new entry
				else {
					currType = new HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>>();
					currFile.put(dataType, currType);
				}

				// Get the viewed LVs for this type
				ViewModel viewModel = model.getViewModel();
				ArrayList<Integer> viewedLvs = viewModel.getViewedLvs(dataType);

				// First remove Lvs which are no longer being viewed
				ArrayList<Integer> lvsToRemove = new ArrayList<Integer>();
				for (Integer lv : currType.keySet() ) {
					if (!viewedLvs.contains(lv)) {
						lvsToRemove.add(lv);
					}
				}
				for (Integer lv : lvsToRemove) {
					currType.remove(lv);
				}

				// For each LV
				for (Integer lv : viewedLvs) {
					// Get the colour scale associated with the curent type and lv number
					double[] colourScale = globalColourScale;

					ColorGradient colourGradient = new ColorGradient(colourScale[0], colourScale[1], colourScale[2]);

					HashMap<Integer, HashMap<Integer, BrainImage>> currLv = null;

					// Obtain the existing entry
					if (currType.containsKey(lv)) {
						currLv = currType.get(lv);
					}
					// Or add a new entry
					else {
						currLv = new HashMap<Integer, HashMap<Integer, BrainImage>>();
						currType.put(lv, currLv);
					}

					// First remove the lags which are no longer being viewed
					ArrayList<Integer> lagsToRemove = new ArrayList<Integer>();
					lagNumbers = mRepository.getControlPanelModel().getViewableLags(resultFile, brainView);
					for (Integer lag : currLv.keySet() ) {
						if (!lagNumbers.contains(lag) ) { 
							lagsToRemove.add(lag);        
						}
					}
					for (Integer lag : lagsToRemove) {
						currLv.remove(lag);
					}

					// For each lag
					for (Integer lag : lagNumbers) { 
						HashMap<Integer, BrainImage> currLag = null;

						// Obtain the existing entry
						if (currLv.containsKey(lag) ) {
							currLag = currLv.get(lag);
						}
						else {
							currLag = new HashMap<Integer, BrainImage>();
							currLv.put(lag, currLag);
						}

						// First remove the slices which are no longer being viewed
						ArrayList<Integer> slicesToRemove = new ArrayList<Integer>();
						for (Integer slice : currLag.keySet() ) {
							if (!sliceNumbers.contains(slice) ) {
								slicesToRemove.add(slice);
							}
						}
						for (Integer slice : slicesToRemove) {
							BrainImage image = currLag.get(slice);
							image.removeMouseListener(this);
							image.removeMouseMotionListener(this);

							currLag.remove(slice);
						}

						// For each slice
						for (Integer slice : sliceNumbers) {
							// If this slice isn't already there
							if (!currLag.containsKey(slice) ) {
								// Create a new image for that slice
								//Slice numbers are 1-based.
								BrainImage image = new BrainImage(model, coords.get(slice - 1),
										resultFile, dataType, lv - 1, lag, slice - 1, brainView,
										colourGradient, scale, rotation, useCrosshair, crosshairTransparency,
										crosshairColor, useLabel, labelTransparency, labelColor);

								currLag.put(slice,image);
								image.addMouseListener(this);
								image.addMouseMotionListener(this);
							}
						}

						if (currLag.size() == 0) {
							currLv.remove(lag);
						}
					}

					if (currLv.size() == 0) {
						currType.remove(lv);
					}
				}

				if (currType.size() == 0) {
					currFile.remove(dataType);
				}
			}

			if (currFile.size() == 0) {
				mBrainImages.remove(resultFile);
			}
		}
	}

	private void drawImagePanels(){
		int brainView = mRepository.getImagePropertiesModel().getBrainView();

		double[] globalColourScale = mRepository.getGlobalColourScale();
		
		ArrayList<String> viewedFiles = new ArrayList<String>(mBrainImages.keySet() );
		Collections.sort(viewedFiles);
		
		int numSlicesPerRow = getNumSlicesPerRow();
		
		for (String resultFile : viewedFiles) {
			HashMap<String, HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>>> currFile = mBrainImages.get(resultFile);
			
			// Make a panel for this volume
			JPanel filePanel = new JPanel();
			filePanel.setBorder(new LineBorder(Color.BLACK, 3) );
			filePanel.setLayout(new BoxLayout(filePanel, BoxLayout.Y_AXIS) );
			
			if (showOtherLabels)
				filePanel.add(new JLabel(resultFile));
			
			ArrayList<String> viewedTypes = new ArrayList<String>(currFile.keySet() );
			Collections.sort(viewedTypes);
			
			for (String dataType : viewedTypes) {
				ArrayList<ArrayList<Integer>> coords = mRepository.getGeneral(resultFile).getBrainData(dataType).getSlices(brainView);
				HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>> currType = currFile.get(dataType);
				
				// Make a panel for this type
				JPanel typePanel = new JPanel();
				typePanel.setBorder(new LineBorder(Color.LIGHT_GRAY, 3) );
				typePanel.setLayout(new BoxLayout(typePanel, BoxLayout.Y_AXIS) );
				
				if (showOtherLabels)
					typePanel.add(new JLabel(dataType));
				
				ArrayList<Integer> viewedLvs = new ArrayList<Integer>(currType.keySet() );
				Collections.sort(viewedLvs);
				
				for (Integer lv : viewedLvs) {
					// Get the colour scale associated with the curent type and lv number
					double[] colourScale = globalColourScale;
					
					if (!mRepository.getUseGlobalScale() ) {
							colourScale = mRepository.getGeneral(resultFile).getBrainData(dataType).getColourScaleModel().getColourScale(lv - 1);
					}
					
					ColorGradient colourGradient = new ColorGradient(colourScale[0], colourScale[1], colourScale[2]);
					
					HashMap<Integer, HashMap<Integer, BrainImage>> currLv = currType.get(lv);
					
					// Make a panel for this LV
					JPanel lvPanel = new JPanel();
					lvPanel.setBorder(new LineBorder(Color.GRAY, 3) );
					lvPanel.setLayout(new BoxLayout(lvPanel, BoxLayout.Y_AXIS) );
					
					
					ResultModel result = mRepository.getGeneral(resultFile);
					String fName = result.getFilename();
					
					boolean isNpairs;
					if (fName.endsWith(NpairsfMRIResultFileFilter.EXTENSION)) 
						isNpairs = true;
					else 
						isNpairs = false;
						
					if (showOtherLabels)
						if (isNpairs)
							lvPanel.add(new JLabel("CV #" + lv) );
						else
							lvPanel.add(new JLabel("LV #" + lv) );
					
					ArrayList<Integer> viewedLags = new ArrayList<Integer>(currLv.keySet() );
					Collections.sort(viewedLags);
					
					ArrayList<ArrayList<JPanel>> panels = new ArrayList<ArrayList<JPanel>>(); 
					
					for (Integer lag : viewedLags) {
						ArrayList<JPanel> lagPanels = new ArrayList<JPanel>();
						
						int x = 0;
						
						HashMap<Integer, BrainImage> currLag = currLv.get(lag);
						
						//Create a row panel for this lag
						JPanel imageRow = getNewImageRowPanel();
						
						ArrayList<Integer> viewedSlices = new ArrayList<Integer>(currLag.keySet() );
						Collections.sort(viewedSlices);
						
						for (Integer slice : viewedSlices) {
							BrainImage image = currLag.get(slice); 
							if (!image.colGrad.equals(colourGradient) ||
									image.brainView != brainView) {
								image.mCoordinates = coords.get(slice - 1);
								image.colGrad = colourGradient;
								image.brainView = brainView;
								image.reInitializeImage();
							}
							imageRow.add(image);
							
							++x;
							if (x == numSlicesPerRow) {
								lagPanels.add(imageRow);
								imageRow = getNewImageRowPanel();
								x = 0;
							}
						}
						
						// Avoid adding empty rows
						if (imageRow.getComponentCount() > 0) {
							lagPanels.add(imageRow);
						}
						
						panels.add(lagPanels);
					}
					
					// Rearrange the rows so that if, for example,
					// we had 3 rows per lag, we get:
					// lag 0 row 1
					// lag 1 row 1
					// lag 2 row 1
					// lag 0 row 2
					// lag 2 row 2
					// etc.
					if (panels.size() > 0) {
						// Work under the assumption that all lags had the same number of slices
						for (int i = 0; i < panels.get(0).size(); ++i) {
							for (int j = 0; j < panels.size(); ++j) {
								lvPanel.add(panels.get(j).get(i));
							}
						}
					}
					typePanel.add(lvPanel);
				}
				filePanel.add(typePanel);
			}
			add(filePanel);
		}
	}
	
	private JPanel getNewImageRowPanel() {
		JPanel imageRow = new JPanel();
		imageRow.setLayout(new BoxLayout(imageRow, BoxLayout.X_AXIS));
		imageRow.setAlignmentX(JPanel.LEFT_ALIGNMENT);
		return imageRow;
	}

	public void removeAllImages() {
		mBrainImages.clear();
	}
	
	public void scaleImages(double newScale) {
		scale = newScale;
		
		for (String resultFile : mBrainImages.keySet() ) {
			HashMap<String, HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>>> fileMap = mBrainImages.get(resultFile);
			for (String dataType : fileMap.keySet() ) {
				HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>> typeMap = fileMap.get(dataType);
				for (Integer lv : typeMap.keySet() ) {
					HashMap<Integer, HashMap<Integer, BrainImage>> lvMap = typeMap.get(lv);
					for (Integer lag : lvMap.keySet() ) {
						HashMap<Integer, BrainImage> lagMap = lvMap.get(lag);
						for (BrainImage slice : lagMap.values() ) {
							slice.scale(newScale);
						}
					}
				}
			}
		}
	}
	
	/**
	 * Sets the rotation of the image to a multiple of 90 degrees.
	 */
	public void rotateImages(int rot90times) {
		rotation = rot90times;
//		rotation %= 4;
		
		for (String resultFile : mBrainImages.keySet() ) {
			HashMap<String, HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>>> fileMap = mBrainImages.get(resultFile);
			for (String dataType : fileMap.keySet() ) {
				HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>> typeMap = fileMap.get(dataType);
				for (Integer lv : typeMap.keySet() ) {
					HashMap<Integer, HashMap<Integer, BrainImage>> lvMap = typeMap.get(lv);
					for (Integer lag : lvMap.keySet() ) {
						HashMap<Integer, BrainImage> lagMap = lvMap.get(lag);
						for (BrainImage slice : lagMap.values() ) {
							slice.rotate(rotation);
						}
					}
				}
			}
		}
	}

	/**
	 * Reinitializes every image.  If the colour scale and st_coords have
	 * changed for that image, they will be updated.
	 */
	public void reInitialize() {
		int brainView = mRepository.getImagePropertiesModel().getBrainView();
		double[] colourScale = mRepository.getGlobalColourScale();
		
		for (String resultFile : mBrainImages.keySet() ) {
			HashMap<String, HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>>> fileMap = mBrainImages.get(resultFile);
			for (String dataType : fileMap.keySet() ) {
				HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>> typeMap = fileMap.get(dataType);
				
				BrainData bData = mRepository.getGeneral(resultFile).getBrainData(dataType);
				ArrayList<ArrayList<Integer>> coords = bData.getSlices(brainView);
				ColourScaleModel colourModel = bData.getColourScaleModel(); 
				for (Integer lv : typeMap.keySet() ) {
					if (!mRepository.getUseGlobalScale() ) {
						colourScale = colourModel.getColourScale(lv - 1);
					}
					
					ColorGradient gradient = new ColorGradient(colourScale[0], colourScale[1], colourScale[2]);
					
					HashMap<Integer, HashMap<Integer, BrainImage>> lvMap = typeMap.get(lv);
					for (Integer lag : lvMap.keySet() ) {
						HashMap<Integer, BrainImage> lagMap = lvMap.get(lag);
						for (BrainImage slice : lagMap.values() ) {
							slice.mCoordinates = coords.get(slice.sliceNum);
							slice.colGrad = gradient;
							slice.reInitializeImage();
						}
					}
				}
			}
		}
	}

	/**
	 * Event handler for when the user clicks on a brain image.  Updates
	 * the selection every time the user clicks.
	 */
	public void mouseClicked(MouseEvent e) {
		BrainImage bImage = (BrainImage)e.getSource();
		
		Point2D sourcePoint = new Point2D.Double(e.getX(), e.getY() );
		Point2D coordinates = bImage.worldToImageCoords(sourcePoint);
		selectPixel(bImage, (int)coordinates.getX(), (int)coordinates.getY() );
	}

	public void mouseEntered(MouseEvent arg0) {
	}

	public void mouseExited(MouseEvent arg0) {
	}

	public void mousePressed(MouseEvent arg0) {
	}

	public void mouseReleased(MouseEvent arg0) {
	}

	/**
	 * Event handler for when the user clicks and drags the mouse over a brain
	 * image.  Updates the selection every time the user moves the mouse with
	 * the button down.
	 */
	public void mouseDragged(MouseEvent e) {
		BrainImage bImage = (BrainImage)e.getSource();
		
		Point2D sourcePoint = new Point2D.Double(e.getX(), e.getY() );
		Point2D coordinates = bImage.worldToImageCoords(sourcePoint);
		selectPixel(bImage, (int)coordinates.getX(), (int)coordinates.getY() );
	}

	public void mouseMoved(MouseEvent arg0) {
	}
	
	private void selectPixel(BrainImage bImage, int xCoord, int yCoord)
	{
		if (currentlySelected != null && currentlySelected != bImage)
		{
			currentlySelected.setSelectedPixel(-1, -1);
//			currentlySelected.reInitializeImage();
			currentlySelected.repaint();
		}
		
		int brainView = mRepository.getImagePropertiesModel().getBrainView();
		
		String resultFilename = bImage.file;
		BrainData bData = bImage.mResultModel.getBrainData();
		int width = bData.getWidth(brainView);
		int height = bData.getHeight(brainView);
		
		// The y-value is flipped since the image was being displayed
		// upside-down. Because of this, the y-value needs to be flipped
		// for retrieving the brain value as well.
		yCoord = height - 1 - yCoord;
		
		// In the sagittal view, the x-value needs to be flipped as well
		// since the original image was also horizontally-flipped.
		if (brainView == BrainData.SAGITTAL) {
			xCoord = width - 1 - xCoord;
		}
		
		int sliceNumber = bImage.sliceNum;
		
		int xSelect = xCoord, ySelect = yCoord, zSelect = sliceNumber;
		
		if (brainView == BrainData.SAGITTAL) {
			xSelect = sliceNumber;
			ySelect = xCoord;
			zSelect = yCoord;
		}
		else if (brainView == BrainData.CORONAL) {
			ySelect = sliceNumber;
			zSelect = yCoord;
		}
		
		// Execute commands
		ResultsCommandManager.selectResultFile(resultFilename);
		ResultsCommandManager.selectBrainData(bImage.type);
		ResultsCommandManager.selectLv(bImage.lvNum);
		ResultsCommandManager.selectVoxel(bImage.lagNum, xSelect + 1, ySelect + 1, zSelect + 1);
	}

	public BufferedImage concatenateImages() {
		int brainView = mRepository.getImagePropertiesModel().getBrainView();
		
		int width = mRepository.getGeneral().getBrainData().getWidth(brainView);
		int height = mRepository.getGeneral().getBrainData().getHeight(brainView);
		int numSlices = mRepository.getGeneral().getBrainData().getNumSlices(brainView);
		
		int numSlicesPerRow = getNumSlicesPerRow();
		int bigWidth = width * Math.min(numSlices, numSlicesPerRow);
		
		BufferedImage bigImage = new BufferedImage(bigWidth, 32, BufferedImage.TYPE_INT_RGB);
		Graphics2D g = bigImage.createGraphics();
		
		int numFiles = 0;
		int numTypes = 0;
		int numLvs = 0;
		int numRows = 0;
		
		int biggestTextWidth = 0;
		
		Font font = new Font("SansSerif", Font.PLAIN, 12);
		g.setFont(font);
		FontMetrics fm = g.getFontMetrics();
		
		// 1st pass: calculate height
		for (String resultFile : mBrainImages.keySet() ) {
			ResultModel model = mRepository.getGeneral(resultFile);
			File file = new File(resultFile);
			
			int l = fm.stringWidth(file.getName() );
			
			if (l > biggestTextWidth) {
				biggestTextWidth = l;
			}
			
			HashMap<String, HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>>> fileMap = mBrainImages.get(resultFile);
			for (String dataType : fileMap.keySet() ) {
				l = fm.stringWidth(dataType);
				
				if (l > biggestTextWidth) {
					biggestTextWidth = l;
				}
				
				HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>> typeMap = fileMap.get(dataType);
				for (Integer lv : typeMap.keySet() ) {
					String lvString = model.getAbbrVariableType() + "#" + lv.toString();
					l = fm.stringWidth(lvString);
					
					if (l > biggestTextWidth) {
						biggestTextWidth = l;
					}
					
					HashMap<Integer, HashMap<Integer, BrainImage>> lvMap = typeMap.get(lv);
					for (Integer lag : lvMap.keySet() ) {
						HashMap<Integer, BrainImage> lagMap = lvMap.get(lag);
						
						int counter = 0;
						
						for (BrainImage slice : lagMap.values() ) {
							++counter;
							
							if (counter == numSlicesPerRow) {
								++numRows;
								counter = 0;
							}
							
						}
						
						if (counter != 0) {
							++numRows;
						}
					}
					
					++numLvs;
				}
				
				++numTypes;
			}
			
			++numFiles;
		}
		
		int stringOffset = 12;
		int textHeight = 20;
		int y = 0;
		
		bigWidth = showOtherLabels ? Math.max(biggestTextWidth, bigWidth) : bigWidth;
		int bigHeight = (height * numRows) + (showOtherLabels ? (textHeight * (numFiles + numTypes + numLvs) ) : 0);
		
		bigImage = new BufferedImage(bigWidth, bigHeight, BufferedImage.TYPE_INT_RGB);
		g = bigImage.createGraphics();
		
		g.setColor(Color.LIGHT_GRAY);
		g.fillRect(0, 0, bigWidth, bigHeight);
		
		g.setColor(Color.BLACK);
		
		int numRowsPerLag = mRepository.getControlPanelModel().getNumRowsPerLag(brainView);
		
		// 2nd pass: actually draw
		ArrayList<String> viewedFiles = new ArrayList<String>(mBrainImages.keySet() );
		Collections.sort(viewedFiles);
		
		for (String resultFile : viewedFiles) {
			ResultModel model = mRepository.getGeneral(resultFile);
			
			if (showOtherLabels) {
				File file = new File(resultFile);
				g.drawString(file.getName(), 0, y + stringOffset);
				y += textHeight;
			}
			
			HashMap<String, HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>>> fileMap = mBrainImages.get(resultFile);
			
			ArrayList<String> viewedTypes = new ArrayList<String>(fileMap.keySet() );
			Collections.sort(viewedTypes);
			
			for (String dataType : viewedTypes) {
				
				if (showOtherLabels) {
					//g.drawString(dataType, 5, y + stringOffset);
					g.drawString(dataType, 0, y + stringOffset);
					y += textHeight;
				}
				
				HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>> typeMap = fileMap.get(dataType);
				
				ArrayList<Integer> viewedLvs = new ArrayList<Integer>(typeMap.keySet() );
				Collections.sort(viewedLvs);
				
				for (Integer lv : viewedLvs) {
					if (showOtherLabels) {
						String lvString = model.getAbbrVariableType() + "#" + lv.toString();
						//g.drawString(lvString, 10, y + stringOffset);
						g.drawString(lvString, 0, y + stringOffset);
						y += textHeight;
					}
					
					HashMap<Integer, HashMap<Integer, BrainImage>> lvMap = typeMap.get(lv);
					
					ArrayList<Integer> viewedLags = new ArrayList<Integer>(lvMap.keySet() );
					Collections.sort(viewedLags);
					
					for (int i = 0; i < numRowsPerLag; ++i) {
						for (Integer lag : viewedLags) {
							int x = 0;
							
							HashMap<Integer, BrainImage> lagMap = lvMap.get(lag);
							
							ArrayList<Integer> slices = new ArrayList<Integer>(lagMap.keySet() );
							Collections.sort(slices);
							
							for (int j = 0; j < numSlicesPerRow; ++j) {
								int sliceIdx = i * numSlicesPerRow + j;
								
								if (sliceIdx >= slices.size() )
									break;
								
								BrainImage image = lagMap.get(slices.get(sliceIdx) );
								
								if (image == null)
									continue;
								
								BufferedImage bi = image.getSaveImage();
								g.drawImage(bi, x, y, width, height, null);
								
								x += width;
							}
							
							y += height;
						}
					}
				}
			}
		}
		
		return bigImage;
	}
}