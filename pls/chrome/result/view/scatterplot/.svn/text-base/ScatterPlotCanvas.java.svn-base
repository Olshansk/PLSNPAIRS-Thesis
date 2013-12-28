package pls.chrome.result.view.scatterplot;
 
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Polygon;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.TreeSet;

import javax.swing.JComponent;
import javax.swing.JOptionPane;

import pls.chrome.result.blvplot.ColorBarPanel;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.ResultModel;
 
@SuppressWarnings("serial")
public class ScatterPlotCanvas extends JComponent implements MouseListener {
    
	//The amount of space (gap) to be shared by the left and right 
	//sides of the plot.
	final int widthPadding  = 60;
	
	//Same but for the top and bottom of this plot.
	final int heightPadding = 40;
	
	//How much extra space from the widthPadding to give to the left hand side
	//in order to compensate for drawing the value labels.
	final int labelRoomWidth = 20;
	
	//Same but for the labels on the bottom of the graph for the x-axis.
	final int labelRoomHeight = 10;
	
    private EnhancedScatterPlot parent;
	private ResultModel mModelA;
    private ResultModel mModelB;
    private String mDataTypeA;
    private String mDataTypeB;
    private int mLvA;
    private int mLvB;
    private int mLagA;
    private int mLagB;
    
    private boolean polygonClosed;
	private boolean createNewPolygon = false;
		
    double min = Double.MAX_VALUE;
    double max = Double.MIN_VALUE;
     
    HashMap<Point.Double, int[]> normalPointIndices;
    HashMap<Point.Double, int[]> filteredPointIndices;
    //HashSet<Double[]> maskPoints = new HashSet<Double[]>(); 
    List<Double[]> maskPoints = new ArrayList<Double[]>();
    
    private GeneralRepository mRepository;
     
    public ScatterPlotCanvas(GeneralRepository repository,
			                 EnhancedScatterPlot parent) {
        mRepository = repository;
		this.parent = parent;
        addMouseListener(this);
    }
     
    public void setInfo(ResultModel modelA, ResultModel modelB,
                        String dataTypeA, String dataTypeB,
                        int lvA, int lvB, int lagA, int lagB) {
        mModelA = modelA;
        mModelB = modelB;
        mDataTypeA = dataTypeA;
        mDataTypeB = dataTypeB;
        mLvA = lvA;
        mLvB = lvB;
        mLagA = lagA;
        mLagB = lagB;
         
        if (mModelA != null && mModelB != null) {
            TreeSet<Integer> coords = new TreeSet<Integer>(mModelA.getCoordinatesSet() );
            coords.retainAll(mModelB.getCoordinatesSet() );
             
            TreeSet<Integer> filteredCoords = new TreeSet<Integer>(mModelA.getFilteredCoordinates() );
            filteredCoords.retainAll(mModelB.getFilteredCoordinates() );
             
            coords.removeAll(filteredCoords); //all cords not filtered/selected.

            ArrayList<HashMap<Integer, double[]>> bDataA = mModelA.getBrainData(mDataTypeA).getAllData();
            ArrayList<HashMap<Integer, double[]>> bDataB = mModelB.getBrainData(mDataTypeB).getAllData();
             
            HashMap<Integer, double[]> dataA = bDataA.get(mLvA);
            HashMap<Integer, double[]> dataB = bDataB.get(mLvB);
             
            normalPointIndices = new HashMap<Point.Double, int[]>();
            filteredPointIndices = new HashMap<Point.Double, int[]>();
             
//            min = Double.MAX_VALUE;
//            max = Double.MIN_VALUE;
             
            double[] mmtA = mModelA.getBrainData(mDataTypeA).getMaxMinThresh(mLvA);
            double[] mmtB = mModelB.getBrainData(mDataTypeB).getMaxMinThresh(mLvB);
             
            max = Math.max(mmtA[0], mmtB[0]);
            min = Math.min(mmtA[1], mmtB[1]); //min of threshold values if using checkbox.
 
            //For all 'coords' if this coord is present in both models, which
			//it should be, create a point with the value of that coord for
			//both models A and B for x,y respectively. Let that point represent
			//a key for the 1D Coord. Both the 'coords' array and the models
			//should be 0 based.
			for (int k : coords) {
				//if either files doesnt have a value for this cord don't plot
				//the value for that cord.
                if (dataA.containsKey(k) && dataB.containsKey(k) ) {
                    double valA = dataA.get(k)[mLagA];
                    double valB = dataB.get(k)[mLagB];
                     
                    Point.Double currPoint = new Point.Double(valA, valB);
                    normalPointIndices.put(currPoint, new int[] {k, k} );
                }
            }

			//Same thing except for filtered coords.
            for (int k : filteredCoords) {
                if (dataA.containsKey(k) && dataB.containsKey(k) ) {
                    double valA = dataA.get(k)[mLagA];
                    double valB = dataB.get(k)[mLagB];
                     
                    Point.Double currPoint = new Point.Double(valA, valB);
                    filteredPointIndices.put(currPoint, new int[] {k, k} );
                }
            }
             
//            System.out.println("max: " + max + " min: " + min);
        }
    }

    
    @Override
    public void paintComponent(Graphics g) {
    	
        if (normalPointIndices != null) {
            BasicStroke normalStroke = new BasicStroke();
                        
            int d = getDimensions();
            
            int[] zero = getZeroCoordinates();
            int zeroX = zero[0];
            int zeroY = zero[1];

            Graphics2D g2 = (Graphics2D)g;
             
            g2.setStroke(normalStroke);
            g2.setColor(Color.white);
            g2.fillRect(zeroX, zeroY, d, d);
             
            //draw the chart borders and the identity line.
            
            g2.setColor(Color.darkGray);
            g2.drawLine(zeroX, zeroY + d, zeroX + d, zeroY); //identity line
            //push this line down by a single pixel because it gets overwritten
            //by a grey line later.
            g2.drawLine(zeroX, zeroY+1, zeroX + d, zeroY+1); //top border
            g2.drawLine(zeroX + d, zeroY, zeroX + d, zeroY + d); //right border
            g2.drawLine(zeroX + d, zeroY + d, zeroX, zeroY + d); //bottom border
            g2.drawLine(zeroX, zeroY + d, zeroX, zeroY); // left border
            
            drawAxis(d, zeroX, zeroY, g2);
             
            g2.setStroke(normalStroke);
             
            drawPoints(d, zeroX, zeroY, g2);
             
            drawPolygon(d,zeroX, zeroY, g2);
        }
    }

    /** 
     * @return the x and y coordinates of the top left corner of this canvas.
     */
    private int[] getZeroCoordinates(){
    	/* we are inserting the scatter plot into the box that
        is this component, so w - d will tell us how much space there is
        between the left side of the scatter plot and the left side of this
        panel. dividing that value by 2 will divide that amount of space
        between the left and right side of the scatter plot so it becomes.
        centred. the same is done for determining the spacing of the 
        height*/
    	
    	int w = getWidth();
    	int h = getHeight();
    	int d = getDimensions();
    	int zeroX = (w - d) / 2 + labelRoomWidth;
        int zeroY = (h - d) / 2 - labelRoomHeight;
        return new int[] {zeroX,zeroY};
    }
    
	/**
	 * Draw the polygon
	 * @param zeroX zero x coordinate location of the scatter plot
	 * @param zeroY zero y coordinate location of the scatter plot
	 * @param g2
	 */
	private void drawPolygon(int curLength, int zeroX, int zeroY, Graphics2D g2) {
		if (!maskPoints.isEmpty()) {
			Polygon currentPolygon = translatePolygon(curLength, zeroX, zeroY);
		    
		    if (currentPolygon.npoints > 2) {
				//If the polygon is closed fill it in with a diff colour.
				if(polygonClosed){
					g2.setColor(new Color(64, 64, 255, 64));
					g2.fillPolygon(currentPolygon);
					g2.setColor(new Color(64, 64, 255, 128));
					g2.drawPolygon(currentPolygon);
				}else{
					g2.setColor(new Color(10, 10, 255, 128));
					g2.fillPolygon(currentPolygon);
					g2.setColor(new Color(10, 10, 255, 192)); 
					g2.drawPolygon(currentPolygon);
				}
		         
		    } else if (currentPolygon.npoints == 2) {
		        g2.setColor(new Color(10, 10, 255, 192) );
		        g2.drawLine(currentPolygon.xpoints[0],
						currentPolygon.ypoints[0],
						currentPolygon.xpoints[1],
						currentPolygon.ypoints[1]);
		    } else if (currentPolygon.npoints == 1) {
		        g2.setColor(new Color(10, 10, 255, 192) );
		        g2.drawOval(currentPolygon.xpoints[0], currentPolygon.ypoints[0], 2, 2);
		    }
		    //currentPolygon.translate(-zeroX, -zeroY);
		}
	}
	/**
	 * The current points that comprise the mask are not absolute locations. 
	 * Instead they are relative percentages that indicate how far away the
	 * point is from the top left corner of the scatter plot. Whenever the
	 * current polygon needs to be drawn we need to transform these relative
	 * positions into concrete absolute ones. Do so with this function.
	 * @param curLength the current length = width of this scatter plot.
	 * @param zeroX location of the 0 x coordinate of this canvas.
	 * @param zeroY location of the 0 y coordinate of this canvas.
	 * @return a polygon created from the relative locations.
	 */
	Polygon translatePolygon(int curLength, int zeroX, int zeroY){
		
		Polygon cPolygon = new Polygon();
		
		for(Double [] relPos : maskPoints){
			double relposx = relPos[0];
			double relposy = relPos[1];
			
			cPolygon.addPoint((int) Math.round(relposx * curLength), 
							(int) Math.round(relposy * curLength));
		}
		cPolygon.translate(zeroX, zeroY);
		return cPolygon;
	}
	
	/**
	 * Draw the points of the scatter plot.
	 * @param d the length of the scatter plot (same as the width since square)
	 * @param zeroX zero x coordinate location of the scatter plot
	 * @param zeroY zero y coordinate location of the scatter plot.
	 * @param g2
	 */
	private void drawPoints(int d, int zeroX, int zeroY, Graphics2D g2) {
		g2.setColor(new Color(192, 0, 0) );
		for (Point.Double pt : normalPointIndices.keySet() ) {
		     
		    Point transformedPoint = transformPoint(pt, zeroX, zeroY, d);
		    g2.fillRect(transformedPoint.x, transformedPoint.y, 1, 1);
		}
		 
		g2.setColor(Color.blue);
		for (Point.Double pt : filteredPointIndices.keySet() ) {
		     
		    Point transformedPoint = transformPoint(pt, zeroX, zeroY, d);
		    g2.fillRect(transformedPoint.x, transformedPoint.y, 1, 1);
		}
	}

    
    /**
     * Draws the scatter plot x and y axis as well as the inner lines of the
     * scatter plot.
     * @param d the length of the scatter plot (same as the width since square)
     * @param zeroX zero x coordinate location of the scatter plot
     * @param zeroY zero y coordinate location of the scatter plot.
     * @param g2
     */
	private void drawAxis(int d, int zeroX, int zeroY, Graphics2D g2) {
		//draw a cross composed of two lines (0,height/2),(width,height/2)
		//and (width/2,height),(width/2,0). these represent coords on the
		//scatter plot itself and not this component which is larger than
		//the area of the scatter plot drawn. 
		
		BasicStroke dashedStroke = new BasicStroke(
		        1, 
		        BasicStroke.CAP_BUTT,
		        BasicStroke.JOIN_BEVEL,
		        0,
		        new float[] {4}, 
		        0
		      );
		
		g2.setStroke(dashedStroke);
        
        double val = 0;
        double lineRatio = (val - min) / (max - min);
        int xPos = (int)(zeroX + lineRatio * d);
        int yPos = (int)(zeroY + (1 - lineRatio) * d);
        
        g2.drawLine(zeroX, yPos, zeroX + d, yPos);
        g2.drawLine(xPos, zeroY, xPos, zeroY + d );
        g2.setColor(Color.black);
        String labelString = "0";
        int xOffset = g2.getFontMetrics().stringWidth(labelString);
        int yOffset = g2.getFontMetrics().getHeight();
        g2.drawString(labelString, xPos - xOffset / 2, zeroY + d + 15);
        g2.drawString(labelString, zeroX - xOffset - 5, yPos + yOffset / 4);
         
         
        val += (max - min) / 10;
        lineRatio = (val - min) / (max - min);
         
        while (lineRatio < 1) {
            xPos = (int)(zeroX + lineRatio * d);
            yPos = (int)(zeroY + (1 - lineRatio) * d);
             
            g2.setColor(Color.lightGray);
            g2.drawLine(zeroX, yPos, zeroX + d, yPos);
            g2.drawLine(xPos, zeroY, xPos, zeroY + d );
             
            g2.setColor(Color.black);
            labelString = ColorBarPanel.formatNumber(val);
            xOffset = g2.getFontMetrics().stringWidth(labelString);
            yOffset = g2.getFontMetrics().getHeight();
            g2.drawString(labelString, xPos - xOffset / 2, zeroY + d + 15);
            g2.drawString(labelString, zeroX - xOffset - 5, yPos + yOffset / 4);
             
            val += (max - min) / 10;
            lineRatio = (val - min) / (max - min);
        }
         
        val = 0;
        val -= (max - min) / 10;
        lineRatio = (val - min) / (max - min);
         
        while (lineRatio > 0) {
            xPos = (int)(zeroX + lineRatio * d);
            yPos = (int)(zeroY + (1 - lineRatio) * d);
             
            g2.setColor(Color.lightGray);
            g2.drawLine(zeroX, yPos, zeroX + d, yPos);
            g2.drawLine(xPos, zeroY, xPos, zeroY + d );
             
            g2.setColor(Color.black);
            labelString = ColorBarPanel.formatNumber(val);
            xOffset = g2.getFontMetrics().stringWidth(labelString);
            yOffset = g2.getFontMetrics().getHeight();
            g2.drawString(labelString, xPos - xOffset / 2, zeroY + d + 15);
            g2.drawString(labelString, zeroX - xOffset - 5, yPos + yOffset / 4);
             
            val -= (max - min) / 10;
            lineRatio = (val - min) / (max - min);
        }
	}
    
	/**
	 * Creates a mask based on the selected ROI.
	 * Preconditions: The polygon is closed and the current polygon is not null.
	 * PostConditions: Nothing if the user hit cancel. Otherwise a new mask
	 * is added, the polygon is no longer closed, the current polygon is null,
	 * and the apply mask button is disabled.
	 */
    public void createMask() {
		String filterName = "";

		while (filterName.equals("")
		   || mRepository.getBrainFilter().filterNames().contains(filterName)) {
			filterName = JOptionPane.showInputDialog("Please provide a "
					+ "unique name for this mask.");
			if (filterName == null) {
				return; //User pressed cancel
			}
		}
		mRepository.addBrainFilter(filterName, getFilteredPoints());
		destroyCurrentPolygon();
    }

    @Override
	/**
	 * Add a point to a polygon or start a new polygon.
	 * Precondition: A new polygon is allowed to be drawn or we are in the
	 * middle of drawing a polygon. i.e polygonClosed != true.
	 * Postcondition: A new polygon is started or a new point is added or
	 * the current polygon is closed and no more points can be added. The apply
	 * mask button is enabled if the polygon is closed.
	 */
	public void mouseClicked(MouseEvent arg) {
    	//System.out.println("Mouse event at ("+arg.getX()+","+arg.getY()+")");
    	
    	//false when cancel button, mask created button, or plot button is hit
    	//this implies that the user may want to create a new mask.
    	//if true just return because no more points can be added.
		if(polygonClosed == true){
			return;
		}
         
        // Scatter plot should always be square
        int d = getDimensions();
        
        int[] zeros = getZeroCoordinates();
        int zeroX = zeros[0];
        int zeroY = zeros[1];
        
        /*Record the relative position of this point wrt the length 
        of the canvas. relposX tells us the distance along the width of the
        canvas. This is a percentage value. For instance if relposX is 60%
        this means that the point arg.getX() is 60% away from the top left
        corner of this canvas. relposY does the same but tells us the % away
        from the top left Y position.*/
        double relposX = (arg.getX() - zeroX) / (d*1.0);
		double relposY = (arg.getY() - zeroY) / (d*1.0);
		
        if (arg.getClickCount() == 2) {
        	if (createNewPolygon)
        	{
				createNewPolygon = false;
				parent.setCancelButtonEnabled(true);
				maskPoints.add(new Double[] {relposX,relposY});
				
			} else if (maskPoints.size() > 2){
				polygonClosed = true;
				parent.setMaskButtonEnabled(true);
        	}
        } else if (arg.getClickCount() == 1) { //add another vertex (point).
            if (!maskPoints.isEmpty()) {
				maskPoints.add(new Double[] {relposX,relposY});
			}
        }
         
        getParent().repaint();
    }

	/**
	 * Create a new set of filtered points.
	 * @return the voxels contained in the drawn polygon. this is derived from
	 * checking which voxels in the already masked set and in the non masked set
	 * belong in the area of this polygon.
	 */
	private TreeSet<Integer> getFilteredPoints() {
        TreeSet<Integer> filter = null;
         
        if (mModelA != null && mModelB != null) {
             
            filter = new TreeSet<Integer>();
             
            // Scatter plot should always be square
            int d = getDimensions();
                         
            int[] zeros = getZeroCoordinates();
            int zeroX = zeros[0];
            int zeroY = zeros[1];

            Polygon currentPolygon = translatePolygon(d,zeroX,zeroY);
			//currentPolygon.translate(zeroX, zeroY);

			//For every point in points not filtered out, if they exist
			//in the polygon drawn, then add corresponding coord value to
			//the filters.

			for (Point.Double p : normalPointIndices.keySet() ) {
				Point pt = transformPoint(p, zeroX, zeroY, d);

				if (currentPolygon.contains(pt) ) {
					int[] coords = normalPointIndices.get(p);
					filter.add(coords[0]);
				}
			}

			for (Point.Double p : filteredPointIndices.keySet() ) {
				Point pt = transformPoint(p, zeroX, zeroY, d);

				if (currentPolygon.contains(pt) ) {
					int[] coords = filteredPointIndices.get(p);
					filter.add(coords[0]);
				}
			}
        }
        return filter;
    }
     
    public Point transformPoint(Point.Double pt, int zeroX, int zeroY, int d) {
        double xRatio = (pt.x - min) / (max - min);
        double yRatio = 1 - (pt.y - min) / (max - min);
         
        return new Point( (int)(zeroX + 10 + xRatio * (d - 20) ),
				        (int)(zeroY + 10 + yRatio * (d - 20) ) );
    }

	/**
	 * 
	 * @return whether we are ready to create a new polygon or we have started
	 * creating one already. (true in the first case, false otherwise).
	 */
	public boolean getCreateNewPolygon(){
		return createNewPolygon;
	}

	/**
	 * Erase the current polygon we are working on.
	 * Disables the "apply mask button"
	 */
	public void destroyCurrentPolygon(){
		createNewPolygon = true;
		polygonClosed = false;
		maskPoints.clear();
		parent.setMaskButtonEnabled(false);
		parent.setCancelButtonEnabled(false);
	}
	
	/**
	 * 
	 * @return The length == width of this scatter plot
	 */
	public int getDimensions(){
        /*we want the scatterplot to be square so we need to take the 
        minimum of the height and width to be the length of the sides.
        for instance, if the available 
        width is less than the available height then we need to make the
        height the same value as the width if we want our scatterplot to
        be square.*/
    	int w = getWidth();
    	int h = getHeight();
    	return Math.min(w - widthPadding, h - heightPadding);
    }
	
    @Override
    public void mouseEntered(MouseEvent arg0) {
    }
 
    @Override
    public void mouseExited(MouseEvent arg0) {
    }
 
    @Override
    public void mousePressed(MouseEvent arg0) {
    }
 
    @Override
    public void mouseReleased(MouseEvent arg0) {
    }
} 