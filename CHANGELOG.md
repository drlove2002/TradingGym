# Changelog

<!--next-version-placeholder-->

## v1.2.0 (2023-05-09)
### Refactor
* **Step:** Return fees and cost from step info ([`f395d95`](https://github.com/drlove2002/TradingGym/commit/f395d95d77e241f7e44ef238784d8e57b1114f91))
* **Step:** Return fees and cost from step info ([`4e5b779`](https://github.com/drlove2002/TradingGym/commit/4e5b77967f0dd20be0cb994e0b69cf7b5ed8a235))
* **Render:** Show figure is optional with env parameter ([`0a61245`](https://github.com/drlove2002/TradingGym/commit/0a61245574fb1141e1d8c6f4d5e5da903ebeb5ab))
* **reward:** Better reward penalty for holding ([`f43b616`](https://github.com/drlove2002/TradingGym/commit/f43b616dc498083582ed7d6874bfb7c0a6604abd))
* **reward:** Fees doesn't effect reward ([`64b23e9`](https://github.com/drlove2002/TradingGym/commit/64b23e92bc11f7064ff225b2c4f7b0d2e6344e8a))
* **step:** Initial money as penalty then illegal move ([`ae4b31a`](https://github.com/drlove2002/TradingGym/commit/ae4b31afd900add2c7913ace4b53246dafef9093))
* **Observation Space:** -INF lower bound ([`bb39b87`](https://github.com/drlove2002/TradingGym/commit/bb39b87fd94f6086a41914fb3bb2e63b9b411954))
* **Reward:** Get high reward on good trade and penalty on holding ([`04006bd`](https://github.com/drlove2002/TradingGym/commit/04006bd981449019e7c7c4f7bdda1bd26f589f1e))
* **Reward:** No total reward. Better reward on hold ([`8a719bc`](https://github.com/drlove2002/TradingGym/commit/8a719bca71451a4434ddd1a65f5f09e1e0ee63bf))
* **Reward:** Return accumulated total reward instead of step reward ([`02ba466`](https://github.com/drlove2002/TradingGym/commit/02ba466e0c095f9b182eb416ceb1469354d8be59))
* **Step:** Change return done into two separate done ([`71e579e`](https://github.com/drlove2002/TradingGym/commit/71e579e0212b7b06685b6867dc6419e74c7d4747))
* **Step Function:** Implementing legal action mask on every step ([`11f8e5c`](https://github.com/drlove2002/TradingGym/commit/11f8e5c56202f8c9b7e3956c7a65fe9d74214550))
* **Reward Function:** Changed for better strategy ([`fca1e04`](https://github.com/drlove2002/TradingGym/commit/fca1e04164f8f8c19a19e4cf934765487e290592))
* **Observation Space:** Added future price for window size ([`5d6043f`](https://github.com/drlove2002/TradingGym/commit/5d6043fe7ef6385fe4f2acdba61e22f6dd7af714))
* **Env:** Added custom feature size ([`ea48de5`](https://github.com/drlove2002/TradingGym/commit/ea48de50b065aaf604f56a558496a0eeb2334c6b))

### Feature
* **Continuous Action space:** Now action space is a regression space ([`cd86c6f`](https://github.com/drlove2002/TradingGym/commit/cd86c6f719aeb68c9691ac8da523e926f54d9ddf))

## v1.1.1 (2023-05-06)
### Fix
* **Env:** Datatype of obs should be float32 and info should be dict ([`fe6ef1b`](https://github.com/drlove2002/TradingGym/commit/fe6ef1b4b9cf6db446afbb23872d0b334e6b154c))

## v1.1.0 (2023-05-06)
### Refactor
* **Reward:** Remove reward from plot/render ([`dbb4143`](https://github.com/drlove2002/TradingGym/commit/dbb414348993d039123e0da9c76ae0fefcfae865))

### Feature
* **StockEnvV1:** New environment for training ([`4c3b3f5`](https://github.com/drlove2002/TradingGym/commit/4c3b3f5247f6c6d4fec9d7dc501eb2583022b00f))

### Fix
* **Setup:** Typo of readme.md ([`1f60b8d`](https://github.com/drlove2002/TradingGym/commit/1f60b8d153dbdaf22ebcc535700ac80377e4a035))
* **Profit Calculation:** Latest profit not getting reset ([`688d38e`](https://github.com/drlove2002/TradingGym/commit/688d38ed7923646051760c187b055a0a61248781))
* **Max share:** Not limiting stock holding ([`6be5f3f`](https://github.com/drlove2002/TradingGym/commit/6be5f3fb0bbb464f76527b120c7c37fc9782e62a))
* **sell reward:** Should be multiplied not divided ([`f9474c1`](https://github.com/drlove2002/TradingGym/commit/f9474c131c3acdd1bcd21bd23cdcac5ee621f27f))
* **Gym Make:** Compatible for MuZero ([`6731737`](https://github.com/drlove2002/TradingGym/commit/67317376df451f254f0b36ac992c6f5cd5500236))
