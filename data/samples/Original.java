import java.util.Scanner;

public class Main {
    // Sum of the first n squares.
    static long sumSquares(int n) {
        long total = 0;
        for (int i = 1; i <= n; i++) {
            total += (long) i * i;
        }
        return total;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        System.out.println(sumSquares(n));
    }
}
