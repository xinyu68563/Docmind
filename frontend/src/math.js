// Model prose is escaped; only KaTeX's trusted renderer can produce markup.
export function renderMath(text,katex){
 const escape=s=>s.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))
 const re=/\\\[([\s\S]*?)\\\]|\\\(([\s\S]*?)\\\)|\$([^$\n]+)\$/g
 let out='',last=0
 for(const m of text.matchAll(re)){
  out+=escape(text.slice(last,m.index))
  try{out+=katex.renderToString(m[1]??m[2]??m[3],{displayMode:m[1]!=null,throwOnError:false,trust:false,maxExpand:200,maxSize:15})}
  catch{out+=escape(m[0])}
  last=m.index+m[0].length
 }
 return out+escape(text.slice(last))
}
