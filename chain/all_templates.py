
cmc_currency_map_api_doc="""
Base URL: https://pro-api.coinmarketcap.com/v1/cryptocurrency/map

CoinMarketCap ID Map

Returns a mapping of all cryptocurrencies to unique CoinMarketCap ids. Per our Best Practices we recommend utilizing CMC ID instead of cryptocurrency symbols to securely identify cryptocurrencies with our other endpoints and in your own application logic. Each cryptocurrency returned includes typical identifiers such as name, symbol, and token_address for flexible mapping to id.

By default this endpoint returns cryptocurrencies that have actively tracked markets on supported exchanges. You may receive a map of all inactive cryptocurrencies by passing listing_status=inactive. You may also receive a map of registered cryptocurrency projects that are listed but do not yet meet methodology requirements to have tracked markets via listing_status=untracked. Please review our methodology documentation for additional details on listing states.

Cryptocurrencies returned include first_historical_data and last_historical_data timestamps to conveniently reference historical date ranges available to query with historical time-series data endpoints. You may also use the aux parameter to only include properties you require to slim down the payload if calling this endpoint frequently.

This endpoint is available on the following API plans:
	Basic
	Hobbyist
	Startup
	Standard
	Professional
	Enterprise

Cache / Update frequency: Mapping data is updated only as needed, every 30 seconds.
Plan credit use: 1 API call credit per request no matter query size.
CMC equivalent pages: No equivalent, this data is only available via API.

PARAMETERS:
symbol:
	Type: string
    Optionally pass a comma-separated list of cryptocurrency symbols to return CoinMarketCap IDs for. If this option is passed, other options will be ignored.

Responses:
200 Success
id: The unique cryptocurrency ID for this cryptocurrency.
rank: The rank of this cryptocurrency.
name: The name of this cryptocurrency.
symbol: The ticker symbol for this cryptocurrency, always in all caps.
slug: The web URL friendly shorthand version of this cryptocurrency name.
platform: Metadata about the parent cryptocurrency platform this cryptocurrency belongs to if it is a token, otherwise null.
RESPONSE SCHEMA Sample
{{
	{
		"data": [
			{
				"id": 1,
				"rank": 1,
				"name": "Bitcoin",
				"symbol": "BTC",
				"slug": "bitcoin",
				"is_active": 1,
				"first_historical_data": "2013-04-28T18:47:21.000Z",
				"last_historical_data": "2020-05-05T20:44:01.000Z",
				"platform": null
			}
              ],
		"status": {
		"timestamp": "2018-06-02T22:51:28.209Z",
		"error_code": 0,
		"error_message": "",
		"elapsed": 10,
		"credit_count": 1
		}
	}
}}
400 Bad request
RESPONSE SCHEMA
{{
	{
		"status": {
		"timestamp": "2018-06-02T22:51:28.209Z",
		"error_code": 400,
		"error_message": "Invalid value for \"id\"",
		"elapsed": 10,
		"credit_count": 0
		}
	}
}}
    
"""

cmc_quote_lastest_api_doc="""
Base URL:https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest

Quotes Latest v2 API Documentation
Returns the latest market quote for 1 or more cryptocurrencies. Use the "convert" option to return market values in multiple fiat and cryptocurrency conversions in the same call.
There is no need to use aux to specify a specific market data, and the returned quote contains all market data.

PARAMETERS:
id: One or more comma-separated cryptocurrency CoinMarketCap IDs. Example: 1,2
slug: Alternatively pass a comma-separated list of cryptocurrency slugs. Example: "bitcoin,ethereum"
symbol: Alternatively pass one or more comma-separated cryptocurrency symbols. Example: "BTC,ETH". At least one "id" or "slug" or "symbol" is required for this request.
convert: Optionally calculate market quotes in up to 120 currencies at once by passing a comma-separated list of cryptocurrency or fiat currency symbols. Each additional convert option beyond the first requires an additional call credit. A list of supported fiat options can be found here. Each conversion is returned in its own "quote" object.
convert_id: Optionally calculate market quotes by CoinMarketCap ID instead of symbol. This option is identical to convert outside of ID format. Ex: convert_id=1,2781 would replace convert=BTC,USD in your query. This parameter cannot be used when convert is used.
aux: Default "num_market_pairs,cmc_rank,date_added,tags,platform,max_supply,circulating_supply,total_supply,is_active,is_fiat". Optionally specify a comma-separated list of supplemental data fields to return. Pass num_market_pairs,cmc_rank,date_added,tags,platform,max_supply,circulating_supply,total_supply,market_cap_by_total_supply,volume_24h_reported,volume_7d,volume_7d_reported,volume_30d,volume_30d_reported,is_active,is_fiat to include all auxiliary fields.

RESPONSE
id: The unique CoinMarketCap ID for this cryptocurrency.
name: The name of this cryptocurrency.
symbol: The ticker symbol for this cryptocurrency.
slug: The web URL friendly shorthand version of this cryptocurrency name.
cmc_rank: The cryptocurrency's CoinMarketCap rank by market cap.
num_market_pairs: The number of active trading pairs available for this cryptocurrency across supported exchanges.
circulating_supply: The approximate number of coins circulating for this cryptocurrency.
total_supply: The approximate total amount of coins in existence right now (minus any coins that have been verifiably burned).
market_cap_by_total_supply: The market cap by total supply. This field is only returned if requested through the aux request parameter.
max_supply: The expected maximum limit of coins ever to be available for this cryptocurrency.
date_added: Timestamp (ISO 8601) of when this cryptocurrency was added to CoinMarketCap.
tags: Array of tags associated with this cryptocurrency. Currently only a mineable tag will be returned if the cryptocurrency is mineable. Additional tags will be returned in the future.
platform: Metadata about the parent cryptocurrency platform this cryptocurrency belongs to if it is a token, otherwise null.
self_reported_circulating_supply: The self reported number of coins circulating for this cryptocurrency.
self_reported_market_cap: The self reported market cap for this cryptocurrency.
quote: A map of market quotes in different currency conversions. The default map included is USD. See the flow Quote Map Instructions.

Quote Map Instructions:
price: Price in the specified currency.
volume_24h: Rolling 24 hour adjusted volume in the specified currency.
volume_change_24h: 24 hour change in the specified currencies volume.
volume_24h_reported: Rolling 24 hour reported volume in the specified currency. This field is only returned if requested through the aux request parameter.
volume_7d: Rolling 7 day adjusted volume in the specified currency. This field is only returned if requested through the aux request parameter.
volume_7d_reported: Rolling 7 day reported volume in the specified currency. This field is only returned if requested through the aux request parameter.
volume_30d: Rolling 30 day adjusted volume in the specified currency. This field is only returned if requested through the aux request parameter.
volume_30d_reported: Rolling 30 day reported volume in the specified currency. This field is only returned if requested through the aux request parameter.
market_cap: Market cap in the specified currency.
market_cap_dominance: Market cap dominance in the specified currency.
fully_diluted_market_cap: Fully diluted market cap in the specified currency.
percent_change_1h: 1 hour change in the specified currency.
percent_change_24h: 24 hour change in the specified currency.
percent_change_7d: 7 day change in the specified currency.
percent_change_30d: 30 day change in the specified currency.
"""

quotes_chain_template="""
Please turn the user input into a fully formed question.
User input: {user_input}
"""

consider_what_is_the_product="""
Question: {original_question}
The question is about the latest market trend for a certain product. Only tell me what is the product's name in the question and end with space? 
"""

api_question_template="""
What is the latest market trend of {product}?
"""

quotes_chain_answer="""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""