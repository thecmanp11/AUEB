
import java.io.*;
import java.text.*;
import java.util.*;
import java.util.Scanner;

public class IntroExample1 {

  public static void main(String[] args) throws java.io.IOException {

	int code;
	String name;
	int salary;

	Scanner scanner = new Scanner(System.in);

	FileWriter outFile = new FileWriter("info.txt", true);
	PrintWriter out = new PrintWriter(outFile);

	while (true) {
		System.out.print("*** Employee's Code: ");
		code = scanner.nextInt();
		if (code==-1) break;
		System.out.print("    Employee's Name: ");
		name = scanner.next();
		System.out.print("    Employee's Salary: ");
		salary = scanner.nextInt();
		out.println(code+","+name+","+salary);
	}

	out.close();
	outFile.close();


  } // end of main
} // end of class IntroExample1

