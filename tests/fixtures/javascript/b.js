function vowels(s) {
  let n = 0;
  for (const c of s.toLowerCase()) {
    if ("aeiou".includes(c)) {
      n++;
    }
  }
  return n;
}
console.log(vowels("Hello World"));
