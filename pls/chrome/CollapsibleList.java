package pls.chrome;

import java.awt.Dimension;
import java.awt.Font;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;

import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.JPanel;

@SuppressWarnings("serial")
public class CollapsibleList extends JPanel {
	private ArrayList<CollapsibleComponent> mItems = new ArrayList<CollapsibleComponent>();
	private boolean mModifyHeaderFont = true;
	
	public CollapsibleList() {
		this(true);
	}
	
	public CollapsibleList(boolean modifyHeaderFont) {
		setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));
		mModifyHeaderFont = modifyHeaderFont;
	}
	
	public CollapsibleComponent addItem(String header, JComponent c) {
		CollapsibleComponent item = new CollapsibleComponent(header, c);
		mItems.add(item);
		add(item);
		
		return item;
	}
	
	public Dimension getPreferredSize() {
		int biggestWidth = 0;
		int totalHeight = 0;
		
		for (JComponent c : mItems) {
			Dimension size = c.getPreferredSize();
			biggestWidth = Math.max(biggestWidth, size.width);
			totalHeight += size.height;
		}
		
		Dimension thisSize = new Dimension(biggestWidth, totalHeight);
		return thisSize;
	}
	
	// This prevents the list items from becoming stretched to fill
	// the list vertically.
	public Dimension getMaximumSize() {
		Dimension max = super.getMaximumSize();
		Dimension pref = getPreferredSize();
		
		return new Dimension(max.width, pref.height);
	}
	
	
	protected class CollapsibleComponent extends JPanel implements ActionListener {
		CollapsibleComponentHeader header;
		JComponent content;
		JLabel state;
		
		public CollapsibleComponent(String headerName, JComponent c) {
			setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS) );
			
			header = new CollapsibleComponentHeader(headerName);
			content = c;
			
			header.setAlignmentX(JButton.LEFT_ALIGNMENT);
			content.setAlignmentX(JComponent.LEFT_ALIGNMENT);
			
			add(header);
			add(content);
			
			Dimension headerMaxSize = header.getMaximumSize();
			headerMaxSize.width = getMaximumSize().width;
			header.setMaximumSize(headerMaxSize);
			
			header.addActionListener(this);
		}

		@Override
		public void actionPerformed(ActionEvent event) {
			setExpanded(!content.isVisible() );
		}
		
		public void setExpanded(boolean expanded) {
			content.setVisible(expanded);
			if (expanded) {
				header.state.setText("^");
			} else {
				header.state.setText("v");
			}
		}
		
		public Dimension getPreferredSize() {
			if (header != null && content != null) {
				int maxWidth = 0;
				int maxHeight = 0;
				
				Dimension headerSize = header.getPreferredSize();
				Dimension contentSize = content.getPreferredSize();
				Dimension superSize = super.getPreferredSize();
				
				maxWidth = Math.max(headerSize.width, contentSize.width);
				maxHeight = Math.max(headerSize.height, contentSize.height);
				
	//			return new Dimension(maxWidth, maxHeight);
				return new Dimension(maxWidth, superSize.height);
			} else {
				return super.getPreferredSize();
			}
		}
		
		public Dimension getMaximumSize() {
			Dimension prefSize = getPreferredSize();
			Dimension superSize = super.getMaximumSize();
			
			return new Dimension(superSize.width, prefSize.height);
		}
		
		
		private class CollapsibleComponentHeader extends JButton {
			JLabel state;
			
			public CollapsibleComponentHeader(String headerName) {
				setLayout(new BoxLayout(this, BoxLayout.X_AXIS) );
				
				JLabel name = new JLabel(headerName);
				
				if (mModifyHeaderFont) {
					Font font = name.getFont();
					// Make the header have a bold and slightly larger font
					name.setFont(new Font(font.getFontName(), Font.BOLD, font.getSize() + 1));
				}
				
				state = new JLabel("^");
				
				add(name);
				add(Box.createHorizontalGlue() );
				add(state);
			}
		}
	}
}
