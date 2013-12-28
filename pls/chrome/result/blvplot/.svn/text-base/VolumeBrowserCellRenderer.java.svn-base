package pls.chrome.result.blvplot;

import java.awt.Component;
import java.net.URL;

import javax.swing.ImageIcon;
import javax.swing.JTree;
import javax.swing.tree.DefaultTreeCellRenderer;

@SuppressWarnings("serial")
public class VolumeBrowserCellRenderer extends DefaultTreeCellRenderer {
	private ImageIcon leafYesIcon = null;
	private ImageIcon leafNoIcon = null;
	
	public VolumeBrowserCellRenderer() {
		URL imgURL = this.getClass().getResource("/images/tree_yes.png");
		leafYesIcon = new ImageIcon(imgURL);
		imgURL = this.getClass().getResource("/images/tree_no.png");
		leafNoIcon = new ImageIcon(imgURL);
	}
	
	public Component getTreeCellRendererComponent(JTree tree,
			Object value, boolean selected, boolean expanded,
			boolean leaf, int row, boolean hasFocus) {
		super.getTreeCellRendererComponent(tree, value, selected, expanded, leaf, row, hasFocus);
		
		if (value instanceof VolumeBrowserLeafNode) {
			VolumeBrowserLeafNode node = (VolumeBrowserLeafNode)value;
			
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
