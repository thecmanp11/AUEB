
import java.io.*;
import java.text.*;
import java.util.*;
import java.util.Scanner;

public class IntroExample2 {

  public static void main(String[] args) throws java.io.IOException {

	int code;
	String name;
	int salary;
	String line;

	BufferedReader in = new BufferedReader(new FileReader("info.txt"));

	while ((line = in.readLine()) != null) {
        System.out.println(line);
	}

	System.out.println();
	in.close();

  } // end of main
} // end of class IntroExample2

