import java.util.Scanner;

public class Main {
    static long squares(int count) {
        long acc = 0;
        int k = 1;
        while (k <= count) {
            acc = acc + (long) k * k;
            k++;
        }
        return acc;
    }

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int count = in.nextInt();
        System.out.println(squares(count));
    }
}
