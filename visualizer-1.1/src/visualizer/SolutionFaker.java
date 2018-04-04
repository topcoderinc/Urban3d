package visualizer;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.PrintWriter;

/*
 * Tool to create a fake solution file easily.
 */
public class SolutionFaker {
	int n = 2048;
	int[] data;
	int id = 0;
	
	public static void main(String[] args) {
		new SolutionFaker().run();
	}

	private void run() {
		data = new int[n*n];
		// sol
		rect(1104, 1312, 20, 20); // TP
		rect(1146, 1329, 30, 16); // half in
		rect(1175, 1300, 20, 20); // in
		rect(1200, 1300, 20, 20); // in
		
		
		try {
			PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter("solution.txt")));
			BuildingIO.writeRasterToRLE("RIC_Tile_055", data, null, n, n, out);
			out.close();
		} 
		catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void rect(int x, int y, int w, int h) {
		id++;
		for (int i = 0; i < w; i++) for (int j = 0; j < h; j++) {
			int pos = (x + i) + (y + j) * n;
			data[pos] = id;
		}		
	}
}
