package pls.test;

import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.JTextPane;

import pls.chrome.CollapsibleList;

@SuppressWarnings("serial")
public class CollapsibleListTest extends JFrame {
	public static void main(String[] args) {
		
		CollapsibleListTest mainFrame = new CollapsibleListTest();
		
		JPanel mainPanel = new JPanel();
		mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.X_AXIS));
//		mainPanel.setAlignmentY(JPanel.TOP_ALIGNMENT);
		
		CollapsibleList leftBar = new CollapsibleList();
		
		JTextPane item1 = new JTextPane();
		item1.setText("In quis libero. Proin in sem at ante vulputate laoreet. Quisque condimentum.");
		leftBar.addItem("Item1", item1);
		
		JTextPane item2 = new JTextPane();
		item2.setText("Nulla nibh urna, mollis ac, semper quis, interdum quis, felis. Pellentesque habitant.");
		leftBar.addItem("Item2", item2);
		
		JPanel item3 = new JPanel();
		item3.setLayout(new BoxLayout(item3, BoxLayout.Y_AXIS) );
		item3.add(new JCheckBox("Some boolean parameter: ") );
		item3.add(new JTextField("Some integer parameter: ") );
		item3.add(new JCheckBox("Another boolean parameter: ") );
		item3.add(new JTextField("Another parameter: ") );
		leftBar.addItem("NPairs Example", item3);
		
		JTextPane item4 = new JTextPane();
		item4.setText("Quisque vehicula tellus nec libero. Cum sociis natoque penatibus et magnis dis.");
		leftBar.addItem("Item4", item4);
		
		String spam = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque blandit. Cras euismod varius eros. Donec vitae felis sed neque iaculis convallis. Praesent enim dui, lobortis sit amet, ornare in, semper sit amet, est. Quisque vestibulum, tellus sed tempus euismod, arcu risus rhoncus sem, eu faucibus libero tellus ac quam. Donec sagittis. Nam eget mi vulputate ipsum tempus consectetur. Nam eget erat. Ut erat risus, blandit auctor, euismod vel, fermentum ac, dui. Sed tellus metus, elementum et, mattis at, facilisis ac, augue. Praesent congue ligula eget tellus. Integer fringilla tortor eget ante. Sed lectus nibh, vestibulum adipiscing, venenatis et, vehicula nec, augue. Proin nulla. Suspendisse quis erat. Quisque id nulla sit amet ante blandit iaculis. Donec et dui. Sed dignissim orci et velit. Nunc placerat nunc eu dolor.";
		
		JTextPane spamTextPane = new JTextPane();
		spamTextPane.setText(spam);
		
		leftBar.setAlignmentY(JPanel.TOP_ALIGNMENT);
		spamTextPane.setAlignmentY(JPanel.TOP_ALIGNMENT);
//		mainPanel.setAlignmentX(JPanel.TOP_ALIGNMENT);
		
		mainPanel.add(leftBar);
		mainPanel.add(spamTextPane);
		
		mainFrame.add(mainPanel);
		mainFrame.setSize(800, 600);
		mainFrame.setVisible(true);
	}
}
