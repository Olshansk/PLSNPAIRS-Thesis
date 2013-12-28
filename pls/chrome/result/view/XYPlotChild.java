package pls.chrome.result.view;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Paint;
import java.awt.Stroke;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;

import org.jfree.chart.axis.AxisSpace;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.PlotRenderingInfo;
import org.jfree.chart.plot.PlotState;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.data.xy.XYDataset;

public class XYPlotChild extends XYPlot {
	double[] vals;
	public boolean valsSet = false;
	
	Graphics2D g2 = null;
	
	public XYPlotChild(XYDataset dataset, ValueAxis domainAxis, ValueAxis rangeAxis, XYItemRenderer renderer) {
		super(dataset, domainAxis, rangeAxis, renderer);
	}
	
	public XYPlotChild() {
		super();
	}
	
	public void setVerticalLines(double[] values) {
		if (values != null) {
			vals = values;
			valsSet = true;
		}
		
	}

	
	/*precondition: setVerticalLineValues called with appropriate parameter */
	public void draw(Graphics2D g2, Rectangle2D area, Point2D p, PlotState ps, PlotRenderingInfo pr) {
		super.draw(g2, area, p, ps, pr);
		
		
		AxisSpace space = calculateAxisSpace(g2, area);
        Rectangle2D r = space.shrink(area, null); //r is the dataArea
        
        double magicHor = 4;
        double magicVer = 3;
        
		r.setRect(r.getX() + magicHor, r.getY() + magicVer, r.getWidth() - magicHor, r.getHeight() -  magicVer*2);
		
		if (valsSet) {
			float dash1[] = {10.0f};
			Stroke stroke = new BasicStroke(1.0f, BasicStroke.CAP_BUTT, 
	                BasicStroke.JOIN_MITER, 
	                10.0f, dash1, 0.0f);
			//Stroke stroke = new BasicStroke(1);
			Paint paint = Color.blue;
			
			for (int i = 0; i < vals.length; i++)
				drawVerticalLine(g2, r, vals[i], stroke, paint);
		}
	}
}
