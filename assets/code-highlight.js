/* Shared local syntax highlighting; highlight.js 11.12.0, BSD-3-Clause. */
'use strict';
(() => {
 const labels={auto:'自動辨識',plaintext:'純文字',python:'Python',javascript:'JavaScript',typescript:'TypeScript',c:'C',cpp:'C++',csharp:'C#',java:'Java',bash:'Shell / Bash',xml:'HTML / XML',css:'CSS',json:'JSON',sql:'SQL',yaml:'YAML',swift:'Swift',matlab:'MATLAB'};
 const aliases={js:'javascript',ts:'typescript',html:'xml',sh:'bash',shell:'bash',cs:'csharp','c++':'cpp',py:'python'};
 const escape=text=>String(text).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
 const rendered=new WeakMap();
 function textOf(node){const clone=node.cloneNode(true);clone.querySelectorAll('br').forEach(br=>br.replaceWith('\n'));return clone.textContent||'';}
 function languageOf(figure){const declared=figure.dataset.language;if(declared)return aliases[declared]||declared;const code=figure.querySelector('pre code');const name=(code?.className.match(/language-([\w+-]+)/)||figure.className.match(/highlight\s+([\w+-]+)/))?.[1];return aliases[name]||name||'auto';}
 function highlight(text,language='auto'){
  if(!window.hljs)return {html:escape(text),language};
  try{const result=language==='auto'?hljs.highlightAuto(text,Object.keys(labels).filter(l=>!['auto','plaintext'].includes(l)&&hljs.getLanguage(l))):hljs.highlight(text,{language:hljs.getLanguage(language)?language:'plaintext',ignoreIllegals:true});return {html:result.value,language:result.language||language};}
  catch{return {html:escape(text),language:'plaintext'};}
 }
 function render(root){root.querySelectorAll('figure.highlight').forEach(figure=>{if(figure.hasAttribute('data-code-rows'))return;const pre=figure.querySelector('.code pre')||figure.querySelector('pre code')||figure.querySelector('pre');if(!pre)return;const text=textOf(pre),language=languageOf(figure),previous=rendered.get(pre);if(previous?.text===text&&previous?.language===language)return;const result=highlight(text,language);pre.innerHTML=result.html;pre.classList.add('hljs');rendered.set(pre,{text,language});figure.dataset.codeCaption=labels[result.language]||result.language;const caption=figure.querySelector('.code-caption');if(caption)caption.textContent=figure.dataset.codeCaption;});}
 function alignRows(figure,pre){
  const table=pre.closest('table');
  if(!table||figure.hasAttribute('data-code-rows'))return;
  const doc=pre.ownerDocument;
  // Split highlighted nodes while retaining token wrappers across newlines.
  function split(node){
   if(node.nodeType===3)return node.textContent.split('\n').map(text=>doc.createTextNode(text));
   const pieces=[node.cloneNode(false)];
   for(const child of node.childNodes){
    split(child).forEach((part,index)=>{if(index)pieces.push(node.cloneNode(false));pieces[pieces.length-1].append(part);});
   }
   return pieces;
  }
  const lines=split(pre);
  if(textOf(pre).endsWith('\n'))lines.pop();
  const body=doc.createElement('tbody');
  lines.forEach((line,index)=>{
   const row=doc.createElement('tr'),gutter=doc.createElement('td'),code=doc.createElement('td');
   gutter.className='gutter';code.className='code';
   const number=doc.createElement('pre');number.textContent=String(index+1);gutter.append(number);
   if(!line.textContent)line.textContent='\u200b';
   code.append(line);row.append(gutter,code);body.append(row);
  });
  table.replaceChildren(body);figure.setAttribute('data-code-rows','');
 }
 window.StudioCode={labels,textOf,languageOf,highlight,render,alignRows};
 if(!document.querySelector('.editor-form'))render(document);
})();
