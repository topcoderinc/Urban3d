/*
 * 3D Building Detector Visualizer and Offline Tester
 * by walrus71
 * 
 * Version history:
 * ================
 * 1.1 (2017.12.) NOT PUBLISHED 
 * 		- Added -large-uncertain switch to handle areas covering 
 * 		  more than one uncertain buildings. Used in final testing.
 * 1.0 (2017.10.09)
 *      - Version at contest launch
 *      - Changed IOU threshold to 0.45
 * 0.5 (2017.09.28)
 *      - Changed MIN_AREA to 100.
 *      - Handling visible/invisible boundaries
 * 0.4 (2017.09.25)
 *      - Bugfix: loading from same directory didn't work
 * 0.3 (2017.09.18)
 *      - Connectedness check
 *      - Uncertain buildings
 * 0.2 (2017.09.06)
 *      - Supports new data file structure
 *      - Added shading to DSM view
 *      - Added building borders
 * 0.1 (2017.08.31)
 *      - Initial version
 */
package visualizer;

import static visualizer.Utils.f;
import static visualizer.Utils.f6;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Container;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.RenderingHints;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.awt.font.FontRenderContext;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.io.FileReader;
import java.io.LineNumberReader;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.Vector;

import javax.imageio.ImageIO;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.SwingUtilities;

public class BuildingVisualizer implements ActionListener, ItemListener, MouseListener {
	public static final double NO_Z_DATA = -32767;
	public static final String OOM_MASK_MARKER = "OOM-MASK:";
	public static final String GT_INSTANCES = "_GTI";	
	public static final String GT_CLASSES = "_GTL";	
	public static final String RGB_FILE = "_RGB";	
	public static final String DSM_FILE = "_DSM";	
	public static final int GT_UNCERTAIN = 65;
	private static final int MIN_AREA = 100;
	
	private static final boolean writeStats = false; //TODO false
	
	private boolean allowLargeUncertainAreas = false;
	private boolean hasGui = true;
	private String dataDir;
	private String[] sceneIds;
	private String currentImageId;
	private MapData currentMapData;
	private int[][] matchingMaskT;
	private int[][] matchingMaskS;
	private String truthPath;
	private String solutionPath;
	private Map<String, Scene> idToScene;
	private double iouThreshold = 0.45;
	
	private double scale; // data size / screen size
	private double x0 = 0, y0 = 0; // x0, y0: TopLeft corner of data is shown here (in screen space)
	
	private JFrame frame;
	private JPanel viewPanel, controlsPanel;
	private JCheckBox showTruthCb, showSolutionCb, showIouCb;
	private JLabel zInfoLabel;
	private JComboBox<String> viewSelectorComboBox;
	private JComboBox<String> imageSelectorComboBox;
	private JTextArea logArea;
	private MapView mapView;
	private Font font = new Font("SansSerif", Font.BOLD, 16);
	
	private Color textColor               = Color.black;
	private Color tpFillSolutionColor     = new Color(255, 255, 255,  80);
	private Color tpBorderSolutionColor   = new Color(150, 255, 150, 200);
	private Color tpFillTruthColor  	  = new Color(255, 255, 255,  50);
	private Color tpBorderTruthColor      = new Color(255, 255, 255, 200); 
	private Color fpFillColor             = new Color(255, 255,   0, 100);
	private Color fpBorderColor           = new Color(255, 255,   0, 200);
	private Color fnFillColor             = new Color(  0, 155, 255, 100);
	private Color fnBorderColor           = new Color(  0, 155, 255, 200);
	private Color uncertainFillColor      = new Color(  0,   0,   0,  80);
	
	private Color invalidColor = new Color(50, 150, 200);
	private int invalidColorI = toRGB(50, 150, 200);
	
	private void run() {
		idToScene = new HashMap<>();
		loadImages();
		try {
			loadBuildings(true);
			if (writeStats) writeStats();
			loadBuildings(false);
		}
		catch (Exception e) {
			log("Error loading buildings");
			e.printStackTrace();
		}
		
		boolean hasTruth = false;
		boolean hasSolution = false;
		for (String id: sceneIds) {
			Scene s = idToScene.get(id);
			if (s.truthBuildings.length > 0) hasTruth = true;
			if (s.solutionBuildings.length > 0) hasSolution = true;
		}
		
		if (!hasTruth || !hasSolution) {
			log("Nothing to score");
			if (hasGui) {
				for (String id: sceneIds) {
					log(id);
				}
			}
		}
		else {
			setInfo("Scoring");
			for (String id: sceneIds) {
				Scene s = idToScene.get(id);
				if (s.solutionBuildings.length == 0) {
					log("  No data in solution for scene " + id);
				}
			}
			String detailsMarker = "Details:";
			log(detailsMarker);
			double fSum = 0;
			int fCnt = 0;
			
			for (String id: sceneIds) {
				Metrics result = score(id);
				if (result != null) {
					result.calculate();
					log(id + "\n"
						+ "  TP       : " + result.tp + "\n"
						+ "  FP       : " + result.fp + "\n"
						+ "  FN       : " + result.fn + "\n"
						+ "  Precision: " + f6(result.precision) + "\n"
						+ "  Recall   : " + f6(result.recall) + "\n"
						+ "  F-score  : " + f6(result.fScore));
					fCnt++;
					fSum += result.fScore;
				}
				else {
					log(id + "\n  - not scored");
				}
			}
			
			String result;
			if (fCnt > 0) {
				double score = fSum / fCnt * 1_000_000;
				result = "\nOverall F-score : " + score; //f6(score);
			}
			else {
				result = "\nOverall F-score : 0";
			}
			
			if (hasGui) { // display final result at the top
				String allText = logArea.getText();
				int pos = allText.indexOf(detailsMarker);
				String s1 = allText.substring(0, pos);
				String s2 = allText.substring(pos);
				allText = s1 + result + "\n\n" + s2;
				logArea.setText(allText);
				logArea.setCaretPosition(0);
				System.out.println(result);
			}
			else {
				log(result);
			}
			clearInfo();
		} // anything to score
		
		// the rest is for UI, not needed for scoring
		if (!hasGui) return;
		
		DefaultComboBoxModel<String> cbm = new DefaultComboBoxModel<>(sceneIds);
		imageSelectorComboBox.setModel(cbm);
		imageSelectorComboBox.setSelectedIndex(0);
		imageSelectorComboBox.addItemListener(this);
		
		loadMap(sceneIds[0]);
		repaintMap();
	}

	// private tool to count buildings and sizes
	private void writeStats() {
		for (String id: idToScene.keySet()) {
			Scene scene = idToScene.get(id);
			Building[] truthBuildings = scene.truthBuildings;
			Map<Building, Set<P2>> truthBuildingPoints = scene.makePointSet(true, true);
			int cnt = 0;
			double area = 0;
			for (Building b: truthBuildings) {
				if (b.isUncertain) continue;
				cnt++;
				Set<P2> points = truthBuildingPoints.get(b);
				area += points.size();
			}
			if (cnt > 0) area /= cnt;
			log(id + "\t" + cnt + "\t" + f(area));
		}
		
	}

	/*
	 * NOTE: There are some changes compared to how the online scorer works,
	 * introduced for final testing, when we have some larger uncertain areas.
	 * These can match more than one solution buildings, and also IOU is 
	 * calculated differently. Changes marked with // FINAL-HACK and applied
	 * if doFinalHack is true.
	 */
	private Metrics score(String id) {
		Metrics ret = new Metrics();
		Scene scene = idToScene.get(id);
		Building[] truthBuildings = scene.truthBuildings;
		Building[] solutionBuildings = scene.solutionBuildings;
		if (truthBuildings.length == 0 && solutionBuildings.length == 0) {
			return null;
		}
		if (truthBuildings.length == 0) {
			ret.fp = solutionBuildings.length;
			return ret;
		}
		if (solutionBuildings.length == 0) {
			ret.fn = truthBuildings.length;
			return ret;
		}
		
		Map<Building, Set<P2>> truthBuildingPoints = scene.makePointSet(true, true);
		Map<Building, Set<P2>> solutionBuildingPoints = scene.makePointSet(false, true);
		int tp = 0;
		int fp = 0;
		int fn = 0;
		for (Building sB: solutionBuildings) {
			if (sB.hasError) {
				continue;
			}
			
			Set<P2> sBpoints = solutionBuildingPoints.get(sB);
			Building bestMatchingT = null;
			double maxScore = 0;
			for (Building tB: truthBuildings) {
				// FINAL-HACK uncertain tB can match any times:
				if (allowLargeUncertainAreas) {
					if (tB.match == Match.TP && !tB.isUncertain) continue; // matched already
				}
				else {
					if (tB.match == Match.TP) continue; // matched already
				}
				
				Set<P2> tBpoints = truthBuildingPoints.get(tB);
				double score = iou(tB, sB, tBpoints, sBpoints);
				if (score > maxScore) {
					maxScore = score;
					bestMatchingT = tB;
				}
				else if (score == maxScore && bestMatchingT != null && tB.label < bestMatchingT.label) {
					bestMatchingT = tB;
				}
			}
			sB.iouScore = maxScore;
			if (maxScore > iouThreshold) {
				if (!bestMatchingT.isUncertain) {
					tp++;
				}
				sB.match = Match.TP;
				bestMatchingT.match = Match.TP;
				sB.matchedWith = bestMatchingT;
				bestMatchingT.matchedWith = sB;
			}
			else {
				fp++;
				sB.match = Match.FP;
			}
		}
		for (Building tP: truthBuildings) {
			if (tP.match == Match.NOTHING) {
				if (!tP.isUncertain) {
					fn++;
				}
				tP.match = Match.FN;
			}
		}
		ret.tp = tp;
		ret.fp = fp;
		ret.fn = fn;
		
		return ret;
	}	
	
	private double iou(Building tB, Building sB, Set<P2> tPoints, Set<P2> sPoints) {
		if (tB.minX > sB.maxX || tB.minY > sB.maxY || tB.maxX < sB.minX || tB.maxY < sB.minY) {
			return 0;
		}
		if (tPoints.isEmpty() || sPoints.isEmpty()) {
			return 0;
		}
		int total = sPoints.size();
		int common = 0;
		for (P2 p: tPoints) {
			if (!p.isOom) {
				total++;
				if (sPoints.contains(p)) common++;
			}
		}
		// FINAL-HACK: for uncertain tB we return the area of sB inside the uncertain region
		if (allowLargeUncertainAreas) {
			if (tB.isUncertain) {
				return (double)common / sPoints.size();
			}
			else {
				return (double)common / (total - common);
			}
		}
		else {
			return (double)common / (total - common);
		}		
	}

	private void setInfo(String message) {
		if (hasGui && mapView != null) {
			mapView.setInfo(message);
		}
	}
	private void clearInfo() {
		if (hasGui && mapView != null) {
			mapView.clearInfo();
		}
	}
	
	private void loadImages() {
		String msg = "Reading images from " + dataDir + " ..."; 
		log(msg);
		setInfo(msg);
		recurseDataDir(new File(dataDir));
		
		sceneIds = idToScene.keySet().toArray(new String[0]);
		Arrays.sort(sceneIds);
		clearInfo();
	}
	
	private void recurseDataDir(File dir) {
		for (File f: dir.listFiles()) {
			if (f.isDirectory()) {
				recurseDataDir(f);
			}
			else {
				String name = f.getName();
				if (name.contains(RGB_FILE) && name.endsWith(".tif")) {
					// JAX_TILE_001_RGB.tif
					String id = name.substring(0, name.indexOf(RGB_FILE));
					Scene scene = new Scene(id);
					idToScene.put(id, scene);
					scene.rgbFile = f;
					String dsmName = name.replace(RGB_FILE, DSM_FILE);
					File dsmFile = new File(dir, dsmName);
					if (dsmFile.exists()) {
						scene.dsmFile = dsmFile;
					}
					else {
						log("Can't find " + dsmFile.getAbsolutePath());
					}
				}
			}
		} // for files in dir
	}
	
	private void loadBuildings(boolean truth) throws Exception {
		String what = truth ? "truth" : "solution";
		String path = truth ? truthPath : solutionPath;
		String msg = "Reading " + what + " buildings from " + path;
		log(msg);
		if (path == null || path.isEmpty()) {
			log("  Path not set");
			return;
		}
		setInfo(msg);
		LineNumberReader lnr = new LineNumberReader(new FileReader(path));
        while (true) {
			String line = lnr.readLine();
			if (line == null) break;
			String id = line;
			Scene scene = idToScene.get(id);
			if (scene == null) { // shouldn't happen
				scene = new Scene(id);
				idToScene.put(id, scene);
			}
			line = lnr.readLine();
			String[] parts = line.split(",");
			int w = Integer.parseInt(parts[0]);
			int h = Integer.parseInt(parts[1]);
			scene.w = w;
			scene.h = h;
			int size = w*h;
			int[] data = new int[size];
			
			line = lnr.readLine();
			int[] rle;
			if (line.startsWith(OOM_MASK_MARKER)) {
				line = line.substring((OOM_MASK_MARKER).length());
				rle = lineToRLE(line);
				scene.oomRLE = rle;
				// there's one more line in this case
				line = lnr.readLine();
			}			
			rle = lineToRLE(line);
			int cnt = 0;
			for (int i = 0; i < rle.length / 2; i++) {
				int label = rle[2*i];
				int run = rle[2*i+1];
				for (int j = 0; j < run; j++) {
					data[cnt++] = label;
				}
			}
			if (truth) {
				scene.truthRLE = rle;
			}
			else {
				scene.solutionRLE = rle;
			}
			
			Map<Integer, Building> labelToBuilding = new HashMap<>();
			for (int i = 0; i < w; i++) for (int j = 0; j < h; j++) {
				int label = data[i + w*j];
				if (label != 0) {
					Building b = labelToBuilding.get(label);
					if (b == null) {
						b = new Building();
						b.label = label;
						labelToBuilding.put(label, b);
						if (label < 0) {
							b.isUncertain = true;
						}
					}
				}
			}
			Building[] bArr = labelToBuilding.values().toArray(new Building[0]);
			if (truth) {
				scene.truthBuildings = bArr;
			}
			else {
				scene.solutionBuildings = bArr;
			}
        }
        lnr.close();
        clearInfo();
	}
    
	private int[] lineToRLE(String line) {
		StringTokenizer st = new StringTokenizer(line, ",");
		int n = st.countTokens();
		int[] rle = new int[n];
		for (int i = 0; i < n / 2; i++) {
			int label = Integer.parseInt(st.nextToken());
			int run = Integer.parseInt(st.nextToken());
			rle[2*i] = label;
			rle[2*i+1] = run;
		}
		return rle;
	}

	public static class P2 {
		public int x;
		public int y;
		public boolean isOom = false; // out of visible part of map

		public P2(int x, int y) {
			this.x = x; this.y = y;
		}
		
		@Override
		public String toString() {
			return x + "," + y;
		}
		
		@Override
		public boolean equals(Object o) {
			if (!(o instanceof P2)) return false;
			P2 p = (P2)o;
			return x == p.x && y == p.y;
		}
		
		@Override
		public int hashCode() {
			return (x << 15) + y;
		}
	}
	
	private boolean checkConnected(Set<P2> ps) {
		int n = ps.size();
		if (n == 0) return true;
		
		int w = 10000;
		Set<Integer> is = new HashSet<>();
		for (P2 p: ps) {
			int i = p.x * w + p.y;
			is.add(i);
		}
		int p = is.iterator().next();
		Queue<Integer> q = new LinkedList<>();
		Set<Integer> added = new HashSet<>();
		q.add(p);
		added.add(p);
		while (!q.isEmpty()) {
			p = q.remove();
			int p2;
			p2 = p + 1;
			if (is.contains(p2) && !added.contains(p2)) {q.add(p2); added.add(p2);}
			p2 = p - 1;
			if (is.contains(p2) && !added.contains(p2)) {q.add(p2); added.add(p2);}
			p2 = p + w;
			if (is.contains(p2) && !added.contains(p2)) {q.add(p2); added.add(p2);}
			p2 = p - w;
			if (is.contains(p2) && !added.contains(p2)) {q.add(p2); added.add(p2);}
		}
		
		return added.size() == n;
	}
	
	public class P3 {
		public double x;
		public double y;
		public double z;
		
		public P3(double x, double y, double z) {
			this.x = x; this.y = y; this.z = z;
		}
		
		public P3 cross(P3 v) {
			return new P3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
		}
		
		public double dot(P3 v) {
			return x * v.x + y * v.y + z * v.z;
		}
		
		public double norm() {
			return Math.sqrt(x*x + y*y + z*z);
		}
		
		@Override
		public String toString() {
			return f(x) + ", " + f(y) + ", " + f(z);
		}
	}
	
	public class Scene {
		public String id;
		public File rgbFile;
		public File dsmFile;
		public int w, h;
		public int[] truthRLE = new int[0];
		public int[] solutionRLE = new int[0];
		public int[] oomRLE = null;
		public Building[] truthBuildings = new Building[0];
		public Building[] solutionBuildings = new Building[0];
		
		public Scene(String id) {
			this.id = id;
		}

		public Map<Building, Set<P2>> makePointSet(boolean truth, boolean checkConnected) {
			Map<Building, Set<P2>> ret = new HashMap<>();
			int[] rle = truth ? truthRLE : solutionRLE;
			Building[] bs = truth ? truthBuildings : solutionBuildings;
			
			int[] data = new int[w*h];
			int[] oomMask = new int[w*h];
			int cnt = 0;
			if (truth && oomRLE != null) {
				for (int i = 0; i < oomRLE.length / 2; i++) {
					int label = oomRLE[2*i];
					int run = oomRLE[2*i + 1];
					for (int j = 0; j < run; j++) {
						oomMask[cnt++] = label;
					}
				}
			}
			
			cnt = 0;
			for (int i = 0; i < rle.length / 2; i++) {
				int label = rle[2*i];
				int run = rle[2*i + 1];
				for (int j = 0; j < run; j++) {
					data[cnt++] = label;
				}
			}
						
			Map<Integer, Building> labelToBuilding = new HashMap<>();
			for (Building b: bs) {
				labelToBuilding.put(b.label, b);
				Set<P2> points = new HashSet<>();
				ret.put(b, points);
			}
			
			for (int i = 0; i < w; i++) for (int j = 0; j < h; j++) {
				int v = data[i + w*j];
				if (v != 0) {
					Building b = labelToBuilding.get(v);
					Set<P2> points = ret.get(b);
					P2 p = new P2(i, j);
					if (truth && oomMask[i + w*j] != 0) {
						p.isOom = true;
					}
					points.add(p);
				}
			}
			for (Building b: bs) {
				b.minX = b.minY = Integer.MAX_VALUE;
				b.maxX = b.maxY = -Integer.MAX_VALUE;
				Set<P2> points = ret.get(b);
				int visibleSize = 0;
				boolean onEdge = false; // tile edge or visible boundary
				for (P2 p: points) {
					b.minX = Math.min(b.minX, p.x);
					b.minY = Math.min(b.minY, p.y);
					b.maxX = Math.max(b.maxX, p.x);
					b.maxY = Math.max(b.maxY, p.y);
					
					if (truth) {
						if (p.isOom || p.x == 0 || p.y == 0 || p.x == w-1 || p.y == h-1) {
							onEdge = true;
						}
						if (!p.isOom){
							visibleSize++;
						}
					}
				}
				if (truth && onEdge && visibleSize < MIN_AREA) {
					b.isUncertain = true;
				}
				
				// verify connectedness of solution
				if (!truth && checkConnected) {
					if (!checkConnected(points)) {
						log("Building " + b.label + " not connected. Top left: " + b.minX + "," + b.minY);
						b.hasError  = true;
					}
				}
			}
			
			return ret;		
		}
	}
	
	public static class Building {
		public int label; // id when read from RLE
		public Match match = Match.NOTHING;
		public Building matchedWith = null;
		public boolean isUncertain = false;
		public boolean hasError = false; // true if non-connected solution
		public double iouScore;
		public int minX;
		public int minY;
		public int maxX;
		public int maxY;
	}
	
	private class Metrics {
		public int tp;
		public int fp;
		public int fn;
		public double precision = 0;
		public double recall = 0;
		public double fScore = 0;
		
		public void calculate() {
			if (tp + fp > 0) precision = (double)tp / (tp + fp);
			if (tp + fn > 0) recall = (double)tp / (tp + fn);
			if (precision + recall > 0) {
				fScore = 2 * precision * recall / (precision + recall);
			}
		}
	}
	
	private class MapData {
		public int W;
		public int H;
		public int[][] pixelsRGB;
		public int[][] pixelsDSM;
		public double[][] rawDSM;
		
		public MapData(int w, int h) {
			W = w; H = h;
			pixelsRGB = new int[W][H];
			pixelsDSM = new int[W][H];
			rawDSM = new double[W][H];
		}
	}
	
	private enum Match {
		NOTHING, TP, FP, FN
	}

	
	/**************************************************************************************************
	 * 
	 *              THINGS BELOW THIS ARE UI-RELATED, NOT NEEDED FOR SCORING
	 * 
	 **************************************************************************************************/
	
	public void setupGUI(int W) {
		if (!hasGui) return;
		
		frame = new JFrame("Building Detector Visualizer");
		int H = W * 2 / 3;
		frame.setSize(W, H);
		frame.setResizable(false);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		Container cp = frame.getContentPane();
		cp.setLayout(new GridBagLayout());
		
		GridBagConstraints c = new GridBagConstraints();
		
		c.fill = GridBagConstraints.BOTH;
		c.gridx = 0;
		c.gridy = 0;
		c.weightx = 2;
		c.weighty = 1;
		viewPanel = new JPanel();
		viewPanel.setPreferredSize(new Dimension(H, H));
		cp.add(viewPanel, c);
		
		c.fill = GridBagConstraints.BOTH;
		c.gridx = 1;
		c.gridy = 0;
		c.weightx = 1;
		controlsPanel = new JPanel();
		cp.add(controlsPanel, c);

		viewPanel.setLayout(new BorderLayout());
		mapView = new MapView();
		viewPanel.add(mapView, BorderLayout.CENTER);
		
		controlsPanel.setLayout(new GridBagLayout());
		GridBagConstraints c2 = new GridBagConstraints();
		int gridY = 0;
		
		showTruthCb = new JCheckBox("Show truth buildings");
		showTruthCb.setSelected(true);
		showTruthCb.addActionListener(this);
		c2.fill = GridBagConstraints.BOTH;
		c2.gridx = 0;
		c2.gridy = gridY++;
		c2.weightx = 1;
		controlsPanel.add(showTruthCb, c2);
		
		showSolutionCb = new JCheckBox("Show solution buildings");
		showSolutionCb.setSelected(true);
		showSolutionCb.addActionListener(this);
		c2.gridy = gridY++;
		controlsPanel.add(showSolutionCb, c2);
		
		showIouCb = new JCheckBox("Show IOU scores");
		showIouCb.setSelected(true);
		showIouCb.addActionListener(this);
		c2.gridy = gridY++;
		controlsPanel.add(showIouCb, c2);
		
		zInfoLabel = new JLabel(" XYZ: ");
		c2.gridy = gridY++;
		controlsPanel.add(zInfoLabel, c2);
		
		viewSelectorComboBox = new JComboBox<>(new String[] {"RGB", "DSM"});
		viewSelectorComboBox.setSelectedIndex(0);
		viewSelectorComboBox.addItemListener(this);
		c2.gridy = gridY++;
		controlsPanel.add(viewSelectorComboBox, c2);
		
		imageSelectorComboBox = new JComboBox<>(new String[] {"..."});
		c2.gridy = gridY++;
		controlsPanel.add(imageSelectorComboBox, c2);
		
		JScrollPane sp = new JScrollPane();
		logArea = new JTextArea("", 10, 20);
		logArea.setFont(new Font("Monospaced", Font.PLAIN, 16));
		logArea.addMouseListener(this);
		sp.getViewport().setView(logArea);
		c2.gridy = gridY++;
		c2.weighty = 10;
		controlsPanel.add(sp, c2);
		
		frame.setVisible(true);
	}
	
    private void loadMap(String id) {
    	if (id.equals(currentImageId)) return;
    	
		Scene scene = idToScene.get(id);
    	if (scene.rgbFile == null) {
			log("Can't find RGB file for scene : " + id);
			return;
		}
		if (scene.dsmFile == null) {
			log("Can't find DSM file for scene : " + id);
			return;
		}
		
		MapData md = null;
		int w = 0;
		int h = 0;
		// load 3-band RGB file, 3*8 bit per pixel
		setInfo("Loading " + scene.rgbFile.getName());
		try { 
			BufferedImage img = ImageIO.read(scene.rgbFile);
			w = img.getWidth();
			h = img.getHeight();
			md = new MapData(w, h);
			for (int i = 0; i < w; i++) for (int j = 0; j < h; j++) {
				md.pixelsRGB[i][j] = img.getRGB(i, j);
			}
		} 
		catch (Exception e) {
			log("Error reading RGB image from " + scene.rgbFile.getAbsolutePath());
			e.printStackTrace();
			return;
		}
		
		// load 1-band DSM file, 1*32 bit floating point
		setInfo("Loading " + scene.dsmFile.getName());		
		try {
			BufferedImage img = ImageIO.read(scene.dsmFile);
			Raster raster = img.getRaster();
			List<Double> dList = new Vector<>(w*h);
			double[] samples = new double[w*h];
			raster.getSamples(0, 0, w, h, 0, samples);
			for (int i = 0; i < w; i++) for (int j = 0; j < h; j++) {
				double data = samples[i + j*w];
				md.rawDSM[i][j] = data;
				if (data != NO_Z_DATA) {
					dList.add(data);
				}
			}
			Double[] dArr = dList.toArray(new Double[0]);
			Arrays.sort(dArr);
			// distribute z values into equal-sized buckets
			int n = heatMap.length;
			double[] buckets = new double[n];
			double r = dArr.length / (double)n;
			for (int i = 0; i < n; i++) {
				int index = (int)(i * r);
				buckets[i] = dArr[index];
			}
			for (int i = 0; i < w; i++) for (int j = 0; j < h; j++) {
				double d = md.rawDSM[i][j];
				if (d == NO_Z_DATA) {
					md.pixelsDSM[i][j] = invalidColorI;
				}
				else {
					int ii = Math.abs(Arrays.binarySearch(buckets, d));
					if (ii < 0) ii = 0;
					if (ii >= n) ii = n-1; 
					md.pixelsDSM[i][j] = heatMap[ii];
				}
			}
		} 
		catch (Exception e) {
			log("Error reading DSM image from " + scene.dsmFile.getAbsolutePath());
			e.printStackTrace();
		}
		currentMapData = md;
		currentImageId = id;
		
		// fill matching masks
		// TP: 2
		// FP,FN: 1
		// border: +4
		// uncertain: +8
		matchingMaskT = new int[w][h];
		Map<Building, Set<P2>> bToPoints = scene.makePointSet(true, false);
		Building[] bs = scene.truthBuildings;
		if (bs != null) {
			for (Building b: bs) {
				int d = b.match == Match.TP ? 2 : 1;
				for (P2 p: bToPoints.get(b)) {
					if (p.x < w && p.y < h) {
						matchingMaskT[p.x][p.y] = d;
						// a hack to get a striped effect for uncertain buildings
						if (b.isUncertain) {
							int cell = (p.x + p.y) / 5;
							if (cell % 2 == 0) {
								matchingMaskT[p.x][p.y] += 8;
							}
						}
					}
				}
			}
		}
		
		matchingMaskS = new int[w][h];
		bToPoints = scene.makePointSet(false, false);
		bs = scene.solutionBuildings;
		if (bs != null) {
			for (Building b: bs) {
				int d = b.match == Match.TP ? 2 : 1;
				for (P2 p: bToPoints.get(b)) {
					if (p.x < w && p.y < h) {
						matchingMaskS[p.x][p.y] = d;
					}
				}
			}
		}
		matchingMaskT = findBorders(matchingMaskT, w, h);
		matchingMaskS = findBorders(matchingMaskS, w, h);
		
		// add shadows to DSM map just for fun
		addShadows(currentMapData);
		
		scale = (double)currentMapData.W / mapView.getWidth(); 
		x0 = 0;
		y0 = 0;
		clearInfo();
	}
	
	private void addShadows(MapData md) {
		int W = md.W;
		int H = md.H;
		P3 sun = new P3(1, -2, 1);
		double sunNorm = sun.norm();
		double[][] data = md.rawDSM;
		for (int i = 1; i < W-1; i++) for (int j = 1; j < H-1; j++) {
			int mapColor = md.pixelsDSM[i][j];
			if (mapColor == invalidColorI) continue;
			
			double[] sumsX = new double[2]; // sum and sumW
			double[] sumsY = new double[2];
			
			addDiff(data, i-1, j-1, true, sumsX, 1);
			addDiff(data, i  , j-1, true, sumsX, 1);
			addDiff(data, i-1, j  , true, sumsX, 2);
			addDiff(data, i  , j  , true, sumsX, 2);
			addDiff(data, i-1, j+1, true, sumsX, 1);
			addDiff(data, i  , j+1, true, sumsX, 1);

			addDiff(data, i-1, j  , false, sumsY, 1);
			addDiff(data, i  , j-1, false, sumsY, 2);
			addDiff(data, i  , j  , false, sumsY, 2);
			addDiff(data, i+1, j-1, false, sumsY, 1);
			addDiff(data, i+1, j  , false, sumsY, 1);

			if (sumsX[1] > 0 && sumsY[1] > 0) {
				double dzx = sumsX[0] / sumsX[1];
				double dzy = sumsY[0] / sumsY[1];
				P3 v1 = new P3(1, 0, dzx);
				P3 v2 = new P3(0, 1, dzy);
				P3 tangent = v1.cross(v2);
				double cos = tangent.dot(sun) / (tangent.norm() * sunNorm);
				int shade = cos < 0 ? 0 : 255;
				double a = Math.abs(cos) * 0.8;
				int shaded = shade(mapColor, shade, a);
				md.pixelsDSM[i][j] = shaded;
			}
		}
	}
	
	private int shade(int c, int shade, double a) {
		int r = (c >> 16) & 0xff;
		int g = (c >>  8) & 0xff;
		int b = (c >>  0) & 0xff;
		r = (int)(r * (1-a) + shade * a);
		g = (int)(g * (1-a) + shade * a);
		b = (int)(b * (1-a) + shade * a);
		return (r << 16) | (g << 8) | b;
	}
	
	private void addDiff(double[][] d, int i, int j, boolean horizontal, double[] sums, double w) {
		double d1 = d[i][j];
		if (d1 == NO_Z_DATA) return;
		double d2 = horizontal ? d[i+1][j] : d[i][j+1];
		if (d2 == NO_Z_DATA) return;
		sums[0] += w * (d2 - d1);
		sums[1] += w;
	}

	private int[][] findBorders(int[][] mask, int w, int h) {
		int[][] ret = new int[w][h];
		for (int i = 0; i < w; i++) for (int j = 0; j < h; j++) ret[i][j] = mask[i][j];
		for (int i = 1; i < w-1; i++) for (int j = 1; j < h-1; j++) {
			int m = mask[i][j];
			if (m == 0) continue;
			if (mask[i-1][j] != m || mask[i+1][j] != m || mask[i][j-1] != m || mask[i][j+1] != m) {
				ret[i][j] = m + 4;
			}
		}		
		return ret;
	}

	private static int[] heatMap;
	static {
		heatMap = new int[4 * 256];
		for (int i = 0; i < 256; i++) {
			heatMap[i + 0*256] = toRGB(0, i, 255); 			// blue   -> cyan
			heatMap[i + 1*256] = toRGB(0, 255, 255 - i); 	// cyan   -> green
			heatMap[i + 2*256] = toRGB(i, 255, 0); 			// green  -> yellow
			heatMap[i + 3*256] = toRGB(255, 255 - i, 0); 	// yellow -> red
		}
	}
	
	private static int toRGB(int r, int g, int b) {
		return (r << 16) | (g << 8) | b;
	}
	
	private void repaintMap() {
		if (mapView != null) {
			mapView.repaint();
			frame.repaint();
		}
	}
	
	@SuppressWarnings("serial")
	private class MapView extends JLabel implements MouseListener, MouseMotionListener, MouseWheelListener {
		
		private int mouseX;
		private int mouseY;
		private BufferedImage image;
		private String message = null;
		
		public MapView() {
			super();
			this.addMouseListener(this);
			this.addMouseMotionListener(this);
			this.addMouseWheelListener(this);
		}		
		
		public void setInfo(String s) {
			message = s;
			this.paintImmediately(0, 0, this.getWidth(), this.getHeight());
		}
		
		public void clearInfo() {
			setInfo(null);
		}

		@Override
		public void paint(Graphics gr) {
			int W = this.getWidth();
			int H = this.getHeight();
			Graphics2D g2 = (Graphics2D) gr;
			g2.setFont(font);
			g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
			g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
			
			if (message != null) {
				g2.setColor(invalidColor);
				g2.fillRect(0, 0, W, H);
				g2.setColor(Color.black);
				g2.drawString(message + " ...", 50, 50);
				return;
			}
			
			if (currentMapData == null) return;
			if (image == null) {
				image = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);
			}
			
			int[][] pixels = viewSelectorComboBox.getSelectedIndex() == 0 ? currentMapData.pixelsRGB : currentMapData.pixelsDSM;
			
			for (int i = 0; i < W; i++) for (int j = 0; j < H; j++) {
				int c = invalidColorI;
				int mapI = (int)((i - x0) * scale);
				int mapJ = (int)((j - y0) * scale);
				
				boolean onScreen = false;
				if (mapI >= 0 && mapJ >= 0 && mapI < currentMapData.W && mapJ < currentMapData.H) {
					onScreen = true;
					c = pixels[mapI][mapJ];
				}
				Color trueMaskColor = null;
				// TP: 2
				// FP,FN: 1
				// border: +4
				// uncertain: +8
				if (onScreen && showTruthCb.isSelected()) {
					int v = matchingMaskT[mapI][mapJ];
					if (v == 1) trueMaskColor = fnFillColor;
					else if (v == 5) trueMaskColor = fnBorderColor;
					else if (v == 2) trueMaskColor = tpFillTruthColor;
					else if (v == 6) trueMaskColor = tpBorderTruthColor;
					else if (v >= 8) trueMaskColor = uncertainFillColor;
				}
				Color solutionMaskColor = null;
				if (onScreen && showSolutionCb.isSelected()) {
					int v = matchingMaskS[mapI][mapJ];
					if (v == 1) solutionMaskColor = fpFillColor;
					else if (v == 5) solutionMaskColor = fpBorderColor;
					else if (v == 2) solutionMaskColor = tpFillSolutionColor;
					else if (v == 6) solutionMaskColor = tpBorderSolutionColor;
					else if (v >= 8) solutionMaskColor = uncertainFillColor;
				}
				if (trueMaskColor != null || solutionMaskColor != null) {
					c = combineColors(c, trueMaskColor, solutionMaskColor);
				}				
				image.setRGB(i, j, c);
			}
			g2.drawImage(image, 0, 0, null);
			
			if (showSolutionCb.isSelected() && showIouCb.isSelected()) {
				g2.setColor(textColor);
				Building[] solutionBuildings = idToScene.get(currentImageId).solutionBuildings;
				if (solutionBuildings != null) {
					for (Building p: solutionBuildings) {
						String label = null;
						label = f(p.iouScore);
						int centerX = (int)((p.maxX + p.minX) / 2 / scale + x0);
						int centerY = (int)((p.maxY + p.minY) / 2 / scale + y0);
						int w = textWidth(label, g2);
						int h = font.getSize();
						g2.drawString(label, centerX - w/2, centerY + h/2);
					}
				}
			}
		}
		
		private int textWidth(String text, Graphics2D g) {
			FontRenderContext context = g.getFontRenderContext();
			Rectangle2D r = font.getStringBounds(text, context);
			return (int) r.getWidth();
		}
		
		private int combineColors(int c, Color c1, Color c2) {
			if (c1 != null) c = combineColors(c, c1);
			if (c2 != null) c = combineColors(c, c2);
			return c;
		}
		
		private int combineColors(int c, Color c1) {
			int r = (c >> 16) & 0xff;
			int g = (c >>  8) & 0xff;
			int b = (c >>  0) & 0xff;
			double a = (double)c1.getAlpha() / 255;
			r = (int)(r * (1-a) + c1.getRed() * a);
			g = (int)(g * (1-a) + c1.getGreen() * a);
			b = (int)(b * (1-a) + c1.getBlue() * a);
			return (r << 16) | (g << 8) | b;
		}

		@Override
		public void mouseClicked(java.awt.event.MouseEvent e) {
			if (SwingUtilities.isRightMouseButton(e)) { // right click
				int mode = viewSelectorComboBox.getSelectedIndex();
				viewSelectorComboBox.setSelectedIndex(1 - mode);
				repaintMap();
			}
		}
		@Override
		public void mouseReleased(java.awt.event.MouseEvent e) {
			repaintMap();
		}
		@Override
		public void mouseEntered(java.awt.event.MouseEvent e) {
			setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));
		}
		@Override
		public void mouseExited(java.awt.event.MouseEvent e) {
			setCursor(Cursor.getDefaultCursor());
		}

		@Override
		public void mousePressed(java.awt.event.MouseEvent e) {
			int x = e.getX();
			int y = e.getY();
			mouseX = x;
			mouseY = y;
			repaintMap();
		}
		
		@Override
		public void mouseDragged(java.awt.event.MouseEvent e) {
			int x = e.getX();
			int y = e.getY();
			x0 += x - mouseX;
			y0 += y - mouseY;
			mouseX = x;
			mouseY = y;
			repaintMap();
		}

		@Override
		public void mouseMoved(java.awt.event.MouseEvent e) {
			if (currentMapData == null) return;
			int x = e.getX();
			int y = e.getY();
			int i = (int)((x - x0) * scale);
			int j = (int)((y - y0) * scale);
			String info = "NA";
			if (i >= 0 && j >= 0 && i < currentMapData.W && j < currentMapData.H) {
				double d = currentMapData.rawDSM[i][j];
				if (d != NO_Z_DATA) {
					info = i + ", " + j + ", " + f(d);
				}
			}
			zInfoLabel.setText(" XYZ: " + info);
		}

		@Override
		public void mouseWheelMoved(MouseWheelEvent e) {
			mouseX = e.getX();
			mouseY = e.getY();
			double dataX = (mouseX - x0) * scale;
			double dataY = (mouseY - y0) * scale;
			
			double change =  Math.pow(2, 0.5);
			if (e.getWheelRotation() > 0) scale *= change;
			if (e.getWheelRotation() < 0) scale /= change;
			
			x0 = mouseX - dataX / scale;
			y0 = mouseY - dataY / scale;
			
			repaintMap();
		}
	} // class MapView
	

	@Override
	public void actionPerformed(ActionEvent e) {
		// check boxes clicked
		repaintMap();
	}

	@Override
	public void itemStateChanged(ItemEvent e) {
		if (e.getStateChange() == ItemEvent.SELECTED) {
			if (e.getSource() == imageSelectorComboBox) {
				// new image selected
				String id = (String)imageSelectorComboBox.getSelectedItem();
				loadMap(id);
			}
			else if (e.getSource() == viewSelectorComboBox) {
				// nothing, repaint				
			}
			repaintMap();
		}
	}	

	@Override
	public void mouseClicked(MouseEvent e) {
		if (e.getSource() != logArea) return;
		try {
			int lineIndex = logArea.getLineOfOffset(logArea.getCaretPosition());
			int start = logArea.getLineStartOffset(lineIndex);
			int end = logArea.getLineEndOffset(lineIndex);
			String line = logArea.getDocument().getText(start, end - start).trim();
			for (int i = 0; i < sceneIds.length; i++) {
				if (sceneIds[i].equals(line)) {
					imageSelectorComboBox.setSelectedIndex(i);
					loadMap(sceneIds[i]);
					repaintMap();
				}
			}
		} 
		catch (Exception ex) {
			ex.printStackTrace();
		}
	}

	@Override
	public void mousePressed(MouseEvent e) {}
	@Override
	public void mouseReleased(MouseEvent e) {}
	@Override
	public void mouseEntered(MouseEvent e) {}
	@Override
	public void mouseExited(MouseEvent e) {}
	
	private void log(String s) {
		if (logArea != null) logArea.append(s + "\n");
		System.out.println(s);
	}
	
	private static void exit(String s) {
		System.out.println(s);
		System.exit(1);
	}

	public static void main(String[] args) throws Exception {
		boolean setDefaults = true;
		int n = args.length;
		for (int i = 0; i < n; i++) { // to change settings easily from Eclipse
			if (args[i].equals("-no-defaults")) setDefaults = false;
		}
		
		// TODO remove
		//args = new String[]{"-tiff2txt", "../data/validation", "../data/validation/truth-validation.txt"};
		
		// spec cases: file conversion
		if (n > 0 && args[0].contains("2")) {
			File in = new File(args[1]);
			String out = args[2];
			if (args[0].equals("-txt2tiff")) {
				BuildingIO.txt2Tiff(in, out);
			}
			else if (args[0].equals("-tiff2txt")) {
				BuildingIO.tiff2Txt(in, out);
			}			
			System.exit(0);
		}
		
		BuildingVisualizer v = new BuildingVisualizer();
		v.hasGui = true;
		int w = 1500;
		
		if (setDefaults) {
			v.hasGui = true;
			w = 1500;
			v.truthPath = null;
			v.solutionPath = null;
			v.dataDir = null;
		}
		else {
			// These are just some default settings for local testing, can be ignored.
			
			// sample data
//			v.dataDir = "../data/example/";
//			v.truthPath = "../data/example/truth-example.txt";
//			v.solutionPath = "../data/example/solution-example.txt";
//			v.dataDir = "../data/tam009-scrubbed-2";
//			v.truthPath = "../data/tam009-scrubbed/truth.txt";
//			v.solutionPath = "../data/tam009-scrubbed/solution.txt";
			
			// training data
			//v.dataDir = "../data/train/";
			//v.truthPath = "../data/train/truth-train.txt";
			//v.solutionPath = "../data/train/solution-train.txt";
			
			// test data
//			v.dataDir = "../data/test/";
//			v.truthPath = "../data/test/truth-test.txt";
//			v.solutionPath = "../submissions/final-testing/09-erotemic/min.txt";
			
			// validation
			v.dataDir = "../data/validation/";
			v.truthPath = "../data/validation/truth-validation.txt";
			v.solutionPath = "../submissions/final-testing/14-peARrr/validation.txt";
			v.allowLargeUncertainAreas = true;
			
		}
		
		for (int i = 0; i < n; i++) {
			if (args[i].equals("-no-gui")) v.hasGui = false;
			if (args[i].equals("-w")) w = Integer.parseInt(args[i+1]);
			if (args[i].equals("-iou-threshold")) v.iouThreshold = Double.parseDouble(args[i+1]);
			if (args[i].equals("-truth")) v.truthPath = args[i+1];
			if (args[i].equals("-solution")) v.solutionPath = args[i+1];
			if (args[i].startsWith("-data-dir")) v.dataDir = args[i+1];
			if (args[i].equals("-large-uncertain")) v.allowLargeUncertainAreas = true;			
		}
		
		if (v.hasGui && (v.dataDir == null || v.dataDir.isEmpty())) {
			exit("Data directory not set or empty.");
		}
		
		v.setupGUI(w);
		v.run();
	}

}
