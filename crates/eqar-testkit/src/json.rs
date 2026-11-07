//! Minimal JSON reader (objects, arrays, strings, numbers, bool, null) for fixtures.
//! Numbers are `f64`; integers up to 2^53 are exact.

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Value>),
    Obj(BTreeMap<String, Value>),
}

impl Value {
    /// Object member; panics with the key name when missing (test helper).
    pub fn get(&self, key: &str) -> &Value {
        match self {
            Value::Obj(m) => m.get(key).unwrap_or_else(|| panic!("missing key {key:?}")),
            _ => panic!("not an object (looking for {key:?})"),
        }
    }
    pub fn has(&self, key: &str) -> bool {
        matches!(self, Value::Obj(m) if m.contains_key(key))
    }
    pub fn f64(&self) -> f64 {
        match self {
            Value::Num(n) => *n,
            v => panic!("not a number: {v:?}"),
        }
    }
    pub fn i64(&self) -> i64 {
        self.f64() as i64
    }
    pub fn str(&self) -> &str {
        match self {
            Value::Str(s) => s,
            v => panic!("not a string: {v:?}"),
        }
    }
    pub fn arr(&self) -> &[Value] {
        match self {
            Value::Arr(a) => a,
            v => panic!("not an array: {v:?}"),
        }
    }
    pub fn f64s(&self) -> Vec<f64> {
        self.arr().iter().map(Value::f64).collect()
    }
}

struct P<'a> {
    s: &'a [u8],
    i: usize,
}

impl P<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn eat(&mut self, c: u8) -> Result<(), String> {
        self.ws();
        if self.s.get(self.i) == Some(&c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected {:?} at byte {}", c as char, self.i))
        }
    }
    fn lit(&mut self, word: &str, v: Value) -> Result<Value, String> {
        if self.s[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            Err(format!("bad literal at byte {}", self.i))
        }
    }
    fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        let mut out = Vec::new();
        loop {
            let c = *self.s.get(self.i).ok_or("unterminated string")?;
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = *self.s.get(self.i).ok_or("bad escape")?;
                    self.i += 1;
                    match e {
                        b'"' | b'\\' | b'/' => out.push(e),
                        b'n' => out.push(b'\n'),
                        b't' => out.push(b'\t'),
                        b'r' => out.push(b'\r'),
                        b'u' => {
                            let h = std::str::from_utf8(
                                self.s.get(self.i..self.i + 4).ok_or("bad \\u")?,
                            )
                            .map_err(|e| e.to_string())?;
                            let cp = u32::from_str_radix(h, 16).map_err(|e| e.to_string())?;
                            self.i += 4;
                            let ch = char::from_u32(cp).ok_or("bad \\u code point")?;
                            let mut b = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut b).as_bytes());
                        }
                        _ => return Err(format!("bad escape at byte {}", self.i)),
                    }
                }
                _ => out.push(c),
            }
        }
        String::from_utf8(out).map_err(|e| e.to_string())
    }
    fn value(&mut self) -> Result<Value, String> {
        self.ws();
        match self.s.get(self.i).copied() {
            None => Err("unexpected end".into()),
            Some(b'{') => {
                self.i += 1;
                let mut m = BTreeMap::new();
                self.ws();
                if self.s.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Value::Obj(m));
                }
                loop {
                    self.ws();
                    let k = self.string()?;
                    self.eat(b':')?;
                    m.insert(k, self.value()?);
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Value::Obj(m));
                        }
                        _ => return Err(format!("expected , or }} at byte {}", self.i)),
                    }
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Value::Arr(a));
                }
                loop {
                    a.push(self.value()?);
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Value::Arr(a));
                        }
                        _ => return Err(format!("expected , or ] at byte {}", self.i)),
                    }
                }
            }
            Some(b'"') => Ok(Value::Str(self.string()?)),
            Some(b't') => self.lit("true", Value::Bool(true)),
            Some(b'f') => self.lit("false", Value::Bool(false)),
            Some(b'n') => self.lit("null", Value::Null),
            Some(_) => {
                let st = self.i;
                while self.i < self.s.len()
                    && matches!(
                        self.s[self.i],
                        b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E'
                    )
                {
                    self.i += 1;
                }
                std::str::from_utf8(&self.s[st..self.i])
                    .map_err(|e| e.to_string())?
                    .parse::<f64>()
                    .map(Value::Num)
                    .map_err(|e| format!("bad number at byte {st}: {e}"))
            }
        }
    }
}

pub fn parse(text: &str) -> Result<Value, String> {
    let mut p = P {
        s: text.as_bytes(),
        i: 0,
    };
    let v = p.value()?;
    p.ws();
    if p.i != p.s.len() {
        return Err(format!("trailing data at byte {}", p.i));
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fixture_shapes() {
        let v = parse(r#" {"a":[1,2.5,-3e2],"b":{"s":"x\"yA"},"t":true,"n":null} "#).unwrap();
        assert_eq!(v.get("a").f64s(), vec![1.0, 2.5, -300.0]);
        assert_eq!(v.get("b").get("s").str(), "x\"yA");
        assert_eq!(*v.get("t"), Value::Bool(true));
        assert!(v.has("n") && !v.has("zz"));
        assert!(parse("[1,]").is_err());
        assert!(parse("{} x").is_err());
    }
}
