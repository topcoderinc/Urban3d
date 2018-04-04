package visualizer;

import static visualizer.BuildingVisualizer.DSM_FILE;
import static visualizer.BuildingVisualizer.GT_CLASSES;
import static visualizer.BuildingVisualizer.GT_INSTANCES;
import static visualizer.BuildingVisualizer.GT_UNCERTAIN;
import static visualizer.BuildingVisualizer.NO_Z_DATA;
import static visualizer.BuildingVisualizer.OOM_MASK_MARKER;

import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.LineNumberReader;
import java.io.PrintWriter;

import javax.imageio.ImageIO;

public class BuildingIO {
	/*
	 * Supports only solution RLEs (no uncertain buildings, no out-of-map areas) 
	 */
	public static void txt2Tiff(File in, String outDir) throws Exception {
		LineNumberReader lnr = new LineNumberReader(new FileReader(in));
        while (true) {
			String line = lnr.readLine();
			if (line == null) break;
			String name = line + GT_INSTANCES + ".tif";
			line = lnr.readLine();
			String[] parts = line.split(",");
			int w = Integer.parseInt(parts[0]);
			int h = Integer.parseInt(parts[1]);
			int size = w*h;
			int[] data = new int[size];
			
			line = lnr.readLine();
			parts = line.split(",");
			int cnt = 0;
			for (int i = 0; i < parts.length / 2; i++) {
				int label = Integer.parseInt(parts[2*i]);
				int run = Integer.parseInt(parts[2*i+1]);
				for (int j = 0; j < run; j++) {
					data[cnt++] = label;
				}
			}
			BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_USHORT_GRAY);
			WritableRaster raster = img.getRaster();
			raster.setSamples(0, 0, w, h, 0, data);
			ImageIO.write(img, "tif", new File(outDir, name));
        }
        lnr.close();
	}
	
	public static void tiff2Txt(File f, String outPath) throws Exception {
		PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(outPath)));
		if (f.isDirectory()) {
			recurseTiffDir(f, out);
		}
		else {
			writeBuildingsToTxt(f, out);
		}
		out.close();
		log("Created " + outPath);
	}

	private static void writeBuildingsToTxt(File f, PrintWriter out) throws Exception {
		String name = f.getName();
		if (name.contains(GT_INSTANCES) && name.endsWith(".tif")) {
			log("Processing " + f.getAbsolutePath());
			// AOI_Jax_01_GTI.tif
			String id = name.substring(0, name.indexOf(GT_INSTANCES));
			// load 1-band GTI file, 1*16 bit int
			BufferedImage img = ImageIO.read(f);
			Raster raster = img.getRaster();
			int w = img.getWidth();
			int h = img.getHeight();
			int size = w*h;
			int[] data = raster.getSamples(0, 0, w, h, 0, new int[size]);
			
			// if there is also a GTL then look for uncertain (65), negate value in raster
			String nameClasses = name.replace(GT_INSTANCES, GT_CLASSES);
			File fClasses = new File(f.getParentFile(), nameClasses);
			if (fClasses.exists()) {
				// load 1-band GTL file, 1*8 bit int
				img = ImageIO.read(fClasses);
				raster = img.getRaster();
				int[] dataL = raster.getSamples(0, 0, w, h, 0, new int[size]);
				for (int i = 0; i < size; i++) {
					if (dataL[i] == GT_UNCERTAIN) {
						data[i] *= -1;
					}
				}
			}
			
			// if there is also a DSM then look for out-of-map areas, create a mask for that
			String nameDsm = name.replace(GT_INSTANCES, DSM_FILE);
			File fDsm = new File(f.getParentFile(), nameDsm);
			int[] oomMask = new int[size];
			boolean hasOom = false;
			if (fDsm.exists()) {
				// load 1-band DSM file, 1*32 bit floating point
				img = ImageIO.read(fDsm);
				raster = img.getRaster();
				double[] samples = new double[w*h];
				raster.getSamples(0, 0, w, h, 0, samples);
				for (int i = 0; i < w; i++) for (int j = 0; j < h; j++) {
					double d = samples[i + j*w];
					if (d == NO_Z_DATA) {
						hasOom = true;
						oomMask[i + j*w] = 1;
					}
				}
			}
			if (!hasOom) {
				oomMask = null;
			}
			
			writeRasterToRLE(id, data, oomMask, w, h, out);
		}
	}
	
	public static void writeRasterToRLE(String id, int[] data, int[] oomMask, int w, int h, PrintWriter out) throws Exception {
		out.println(id);
		out.println(w + "," + h);
		int size = w*h;
		int prev, cnt;
		
		if (oomMask != null) {
			// MARKER,{1/0},{run},...\n
			out.print(OOM_MASK_MARKER);
			prev = oomMask[0];
			cnt = 0;
			for (int i = 0; i < size; i++) {
				int d = oomMask[i];
				if (d == prev) {
					cnt++;
				}
				else {
					out.print(prev);
					out.print(",");
					out.print(cnt);
					out.print(",");
					prev = d;
					cnt = 1;
				}
			}
			// last one
			out.print(prev);
			out.print(",");
			out.print(cnt);
			out.println("");
		}
		
		prev = data[0];
		cnt = 0;
		for (int i = 0; i < size; i++) {
			int d = data[i];
			if (d == prev) {
				cnt++;
			}
			else {
				out.print(prev);
				out.print(",");
				out.print(cnt);
				out.print(",");
				prev = d;
				cnt = 1;
			}
		}
		// last one
		out.print(prev);
		out.print(",");
		out.print(cnt);
		out.println("");
	}

	private static void recurseTiffDir(File dir, PrintWriter out) throws Exception {
		for (File f: dir.listFiles()) {
			if (f.isDirectory()) {
				recurseTiffDir(f, out);
			}
			else {
				writeBuildingsToTxt(f, out);
			}
		}
	}
	
	private static void log(String s) {
		System.out.println(s);
	}
	
	// test only
	public static void main(String[] args) throws Exception {
		// create a truth file from a dir of tiffs
//		File f = new File("../data/tam009");
//		tiff2Txt(f, "../data/tmp/truth.txt");
		File f = new File("../data/test");
		tiff2Txt(f, "../data/test/truth-test.txt");

		// test inverse conversion
//		txt2Tiff(new File("../data/tmp/truth.txt"), "../data/tmp");
		
	}
}
