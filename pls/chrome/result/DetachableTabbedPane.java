package pls.chrome.result;

import java.awt.Color;
import java.awt.Component;
import java.awt.Graphics;
import java.awt.Rectangle;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;

import javax.swing.Icon;
import javax.swing.ImageIcon;
import javax.swing.JTabbedPane;

import pls.chrome.result.model.PlotManager;
import pls.chrome.result.view.AttachDetachOption;
import pls.chrome.result.view.RegressionPlot;

@SuppressWarnings("serial")
public class DetachableTabbedPane extends JTabbedPane implements MouseListener {
	
	private PlotManager mPlotManager;
	
	public DetachableTabbedPane() {
		super();

		addMouseListener(this);
	}
	
	public void setPlotManager(PlotManager plotManager) {
		mPlotManager = plotManager;
	}
	
	@Override
	public void addTab(String title, Component component) {
		DetachTabIcon detachIcon = new DetachTabIcon();
		
		super.addTab(title, detachIcon, component);
	}

	@Override
	public void mouseClicked(MouseEvent e) {
		int tabNumber = getUI().tabForCoordinate(this, e.getX(), e.getY() );
		
		if (tabNumber >= 0) {
			Rectangle rect = ((DetachTabIcon)getIconAt(tabNumber)).getBounds();
			
			if (rect.contains(e.getX(), e.getY() ) ) {
				AttachDetachOption component = (AttachDetachOption) 
						getComponentAt(tabNumber);
				mPlotManager.detachPlot(component);

				if(component instanceof RegressionPlot){
					((RegressionPlot) component).setcurDetached(true);
				}
			}
		}
	}

	@Override
	public void mouseEntered(MouseEvent e) {}

	@Override
	public void mouseExited(MouseEvent e) {}

	@Override
	public void mousePressed(MouseEvent e) {}

	@Override
	public void mouseReleased(MouseEvent e) {}

	private class DetachTabIcon implements Icon {
		Icon detachIcon = new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Export16.gif"));
		
		int x_pos;
		int y_pos;
		
		@Override
		public int getIconHeight() {
			return detachIcon.getIconHeight();
		}

		@Override
		public int getIconWidth() {
			return detachIcon.getIconWidth();
		}

		@Override
		public void paintIcon(Component c, Graphics g, int x, int y) {
			x_pos = x;
			y_pos = y;
			
			Color col = g.getColor();
			
			g.setColor(Color.BLACK);
			g.drawLine(x+1, y, x+getIconWidth(), y);
			g.drawLine(x+1, y+getIconHeight(), x+getIconWidth(), y+getIconHeight() );
			g.drawLine(x+1, y, x+1, y+getIconHeight() );
			g.drawLine(x+getIconWidth(), y, x+getIconWidth(), y+getIconHeight() );
			g.setColor(col);
			
			detachIcon.paintIcon(c, g, x+1, y+1);
		}
		
		public Rectangle getBounds() {
			return new Rectangle(x_pos, y_pos, getIconWidth(), getIconHeight() );
		}
	}
}
