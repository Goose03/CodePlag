// Count vowels in a string.
function countVowels(text) {
  let count = 0;
  for (const ch of text.toLowerCase()) {
    if ("aeiou".includes(ch)) {
      count += 1;
    }
  }
  return count;
}

console.log(countVowels("Hello World"));
