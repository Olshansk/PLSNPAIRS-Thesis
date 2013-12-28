package pls.chrome.result.blvplot;

import java.awt.Component;
import java.net.URL;

import javax.swing.DefaultListCellRenderer;
import javax.swing.ImageIcon;
import javax.swing.JList;

@SuppressWarnings("serial")
public class FilterBrowserCellRenderer extends DefaultListCellRenderer {
	private ImageIcon leafYesIcon = null;
	private ImageIcon leafNoIcon = null;
	
	public FilterBrowserCellRenderer() {
		URL imgURL = this.getClass().getResource("/images/tree_yes.png");
		leafYesIcon = new ImageIcon(imgURL);
		imgURL = this.getClass().getResource("/images/tree_no.png");
		leafNoIcon = new ImageIcon(imgURL);
	}
	
	public Component getListCellRendererComponent(JList list,
			Object value, int index, boolean isSelected, boolean cellHasFocus) {
		super.getListCellRendererComponent(list, value, index, isSelected, cellHasFocus);
		
		if (value instanceof FilterBrowserItem) {
			FilterBrowserItem node = (FilterBrowserItem)value;
			
			if (node.beingDisplayed) {
				setIcon(leafYesIcon);
			}
			else {
				setIcon(leafNoIcon);
			}
		}
		
		return this;
	}
}
