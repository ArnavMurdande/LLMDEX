(()=>{var FM=Object.create;var mv=Object.defineProperty;var zM=Object.getOwnPropertyDescriptor;var HM=Object.getOwnPropertyNames;var GM=Object.getPrototypeOf,VM=Object.prototype.hasOwnProperty;var Qi=(t,e)=>()=>{try{return e||t((e={exports:{}}).exports,e),e.exports}catch(n){throw e=0,n}};var kM=(t,e,n,i)=>{if(e&&typeof e=="object"||typeof e=="function")for(let s of HM(e))!VM.call(t,s)&&s!==n&&mv(t,s,{get:()=>e[s],enumerable:!(i=zM(e,s))||i.enumerable});return t};var Gr=(t,e,n)=>(n=t!=null?FM(GM(t)):{},kM(e||!t||!t.__esModule?mv(n,"default",{value:t,enumerable:!0}):n,t));var Cv=Qi(Be=>{"use strict";var Rd=Symbol.for("react.transitional.element"),WM=Symbol.for("react.portal"),XM=Symbol.for("react.fragment"),YM=Symbol.for("react.strict_mode"),qM=Symbol.for("react.profiler"),QM=Symbol.for("react.consumer"),ZM=Symbol.for("react.context"),KM=Symbol.for("react.forward_ref"),JM=Symbol.for("react.suspense"),jM=Symbol.for("react.memo"),_v=Symbol.for("react.lazy"),$M=Symbol.for("react.activity"),eE=Symbol.for("react.view_transition"),gv=Symbol.iterator;function tE(t){return t===null||typeof t!="object"?null:(t=gv&&t[gv]||t["@@iterator"],typeof t=="function"?t:null)}var Sv={isMounted:function(){return!1},enqueueForceUpdate:function(){},enqueueReplaceState:function(){},enqueueSetState:function(){}},Av=Object.assign,Mv={};function Ra(t,e,n){this.props=t,this.context=e,this.refs=Mv,this.updater=n||Sv}Ra.prototype.isReactComponent={};Ra.prototype.setState=function(t,e){if(typeof t!="object"&&typeof t!="function"&&t!=null)throw Error("takes an object of state variables to update or a function which returns an object of state variables.");this.updater.enqueueSetState(this,t,e,"setState")};Ra.prototype.forceUpdate=function(t){this.updater.enqueueForceUpdate(this,t,"forceUpdate")};function Ev(){}Ev.prototype=Ra.prototype;function Dd(t,e,n){this.props=t,this.context=e,this.refs=Mv,this.updater=n||Sv}var Ud=Dd.prototype=new Ev;Ud.constructor=Dd;Av(Ud,Ra.prototype);Ud.isPureReactComponent=!0;var vv=Array.isArray;function Cd(){}var Mt={H:null,A:null,T:null,S:null},Tv=Object.prototype.hasOwnProperty;function Id(t,e,n){var i=n.ref;return{$$typeof:Rd,type:t,key:e,ref:i!==void 0?i:null,props:n}}function nE(t,e){return Id(t.type,e,t.props)}function Bd(t){return typeof t=="object"&&t!==null&&t.$$typeof===Rd}function iE(t){var e={"=":"=0",":":"=2"};return"$"+t.replace(/[=:]/g,function(n){return e[n]})}var xv=/\/+/g;function wd(t,e){return typeof t=="object"&&t!==null&&t.key!=null?iE(""+t.key):e.toString(36)}function sE(t){switch(t.status){case"fulfilled":return t.value;case"rejected":throw t.reason;default:switch(typeof t.status=="string"?t.then(Cd,Cd):(t.status="pending",t.then(function(e){t.status==="pending"&&(t.status="fulfilled",t.value=e)},function(e){t.status==="pending"&&(t.status="rejected",t.reason=e)})),t.status){case"fulfilled":return t.value;case"rejected":throw t.reason}}throw t}function Ca(t,e,n,i,s){var r=typeof t;(r==="undefined"||r==="boolean")&&(t=null);var a=!1;if(t===null)a=!0;else switch(r){case"bigint":case"string":case"number":a=!0;break;case"object":switch(t.$$typeof){case Rd:case WM:a=!0;break;case _v:return a=t._init,Ca(a(t._payload),e,n,i,s)}}if(a)return s=s(t),a=i===""?"."+wd(t,0):i,vv(s)?(n="",a!=null&&(n=a.replace(xv,"$&/")+"/"),Ca(s,e,n,"",function(c){return c})):s!=null&&(Bd(s)&&(s=nE(s,n+(s.key==null||t&&t.key===s.key?"":(""+s.key).replace(xv,"$&/")+"/")+a)),e.push(s)),1;a=0;var o=i===""?".":i+":";if(vv(t))for(var l=0;l<t.length;l++)i=t[l],r=o+wd(i,l),a+=Ca(i,e,n,r,s);else if(l=tE(t),typeof l=="function")for(t=l.call(t),l=0;!(i=t.next()).done;)i=i.value,r=o+wd(i,l++),a+=Ca(i,e,n,r,s);else if(r==="object"){if(typeof t.then=="function")return Ca(sE(t),e,n,i,s);throw e=String(t),Error("Objects are not valid as a React child (found: "+(e==="[object Object]"?"object with keys {"+Object.keys(t).join(", ")+"}":e)+"). If you meant to render a collection of children, use an array instead.")}return a}function Gc(t,e,n){if(t==null)return t;var i=[],s=0;return Ca(t,i,"","",function(r){return e.call(n,r,s++)}),i}function rE(t){if(t._status===-1){var e=t._result,n=e();n.then(function(i){(t._status===0||t._status===-1)&&(t._status=1,t._result=i,n.status===void 0&&(n.status="fulfilled",n.value=i))},function(i){(t._status===0||t._status===-1)&&(t._status=2,t._result=i,n.status===void 0&&(n.status="rejected",n.reason=i))}),t._status===-1&&(t._status=0,t._result=n)}if(t._status===1)return t._result.default;throw t._result}var yv=typeof reportError=="function"?reportError:function(t){if(typeof window=="object"&&typeof window.ErrorEvent=="function"){var e=new window.ErrorEvent("error",{bubbles:!0,cancelable:!0,message:typeof t=="object"&&t!==null&&typeof t.message=="string"?String(t.message):String(t),error:t});if(!window.dispatchEvent(e))return}else if(typeof process=="object"&&typeof process.emit=="function"){process.emit("uncaughtException",t);return}console.error(t)};function bv(t){var e=Mt.T,n={};n.types=e!==null?e.types:null,Mt.T=n;try{var i=t(),s=Mt.S;s!==null&&s(n,i),typeof i=="object"&&i!==null&&typeof i.then=="function"&&i.then(Cd,yv)}catch(r){yv(r)}finally{e!==null&&n.types!==null&&(e.types=n.types),Mt.T=e}}function wv(t){var e=Mt.T;if(e!==null){var n=e.types;n===null?e.types=[t]:n.indexOf(t)===-1&&n.push(t)}else bv(wv.bind(null,t))}var aE={map:Gc,forEach:function(t,e,n){Gc(t,function(){e.apply(this,arguments)},n)},count:function(t){var e=0;return Gc(t,function(){e++}),e},toArray:function(t){return Gc(t,function(e){return e})||[]},only:function(t){if(!Bd(t))throw Error("React.Children.only expected to receive a single React element child.");return t}};Be.Activity=$M;Be.Children=aE;Be.Component=Ra;Be.Fragment=XM;Be.Profiler=qM;Be.PureComponent=Dd;Be.StrictMode=YM;Be.Suspense=JM;Be.ViewTransition=eE;Be.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE=Mt;Be.__COMPILER_RUNTIME={__proto__:null,c:function(t){return Mt.H.useMemoCache(t)}};Be.addTransitionType=wv;Be.cache=function(t){return function(){return t.apply(null,arguments)}};Be.cacheSignal=function(){return null};Be.cloneElement=function(t,e,n){if(t==null)throw Error("The argument must be a React element, but you passed "+t+".");var i=Av({},t.props),s=t.key;if(e!=null)for(r in e.key!==void 0&&(s=""+e.key),e)!Tv.call(e,r)||r==="key"||r==="__self"||r==="__source"||r==="ref"&&e.ref===void 0||(i[r]=e[r]);var r=arguments.length-2;if(r===1)i.children=n;else if(1<r){for(var a=Array(r),o=0;o<r;o++)a[o]=arguments[o+2];i.children=a}return Id(t.type,s,i)};Be.createContext=function(t){return t={$$typeof:ZM,_currentValue:t,_currentValue2:t,_threadCount:0,Provider:null,Consumer:null},t.Provider=t,t.Consumer={$$typeof:QM,_context:t},t};Be.createElement=function(t,e,n){var i,s={},r=null;if(e!=null)for(i in e.key!==void 0&&(r=""+e.key),e)Tv.call(e,i)&&i!=="key"&&i!=="__self"&&i!=="__source"&&(s[i]=e[i]);var a=arguments.length-2;if(a===1)s.children=n;else if(1<a){for(var o=Array(a),l=0;l<a;l++)o[l]=arguments[l+2];s.children=o}if(t&&t.defaultProps)for(i in a=t.defaultProps,a)s[i]===void 0&&(s[i]=a[i]);return Id(t,r,s)};Be.createRef=function(){return{current:null}};Be.forwardRef=function(t){return{$$typeof:KM,render:t}};Be.isValidElement=Bd;Be.lazy=function(t){return{$$typeof:_v,_payload:{_status:-1,_result:t},_init:rE}};Be.memo=function(t,e){return{$$typeof:jM,type:t,compare:e===void 0?null:e}};Be.startTransition=bv;Be.unstable_useCacheRefresh=function(){return Mt.H.useCacheRefresh()};Be.use=function(t){return Mt.H.use(t)};Be.useActionState=function(t,e,n){return Mt.H.useActionState(t,e,n)};Be.useCallback=function(t,e){return Mt.H.useCallback(t,e)};Be.useContext=function(t){return Mt.H.useContext(t)};Be.useDebugValue=function(){};Be.useDeferredValue=function(t,e){return Mt.H.useDeferredValue(t,e)};Be.useEffect=function(t,e){return Mt.H.useEffect(t,e)};Be.useEffectEvent=function(t){return Mt.H.useEffectEvent(t)};Be.useId=function(){return Mt.H.useId()};Be.useImperativeHandle=function(t,e,n){return Mt.H.useImperativeHandle(t,e,n)};Be.useInsertionEffect=function(t,e){return Mt.H.useInsertionEffect(t,e)};Be.useLayoutEffect=function(t,e){return Mt.H.useLayoutEffect(t,e)};Be.useMemo=function(t,e){return Mt.H.useMemo(t,e)};Be.useOptimistic=function(t,e){return Mt.H.useOptimistic(t,e)};Be.useReducer=function(t,e,n){return Mt.H.useReducer(t,e,n)};Be.useRef=function(t){return Mt.H.useRef(t)};Be.useState=function(t){return Mt.H.useState(t)};Be.useSyncExternalStore=function(t,e,n){return Mt.H.useSyncExternalStore(t,e,n)};Be.useTransition=function(){return Mt.H.useTransition()};Be.version="19.3.0"});var tl=Qi((W3,Rv)=>{"use strict";Rv.exports=Cv()});var zv=Qi(It=>{"use strict";function Od(t,e){var n=t.length;t.push(e);e:for(;0<n;){var i=n-1>>>1,s=t[i];if(0<Vc(s,e))t[i]=e,t[n]=s,n=i;else break e}}function Zi(t){return t.length===0?null:t[0]}function Wc(t){if(t.length===0)return null;var e=t[0],n=t.pop();if(n!==e){t[0]=n;e:for(var i=0,s=t.length,r=s>>>1;i<r;){var a=2*(i+1)-1,o=t[a],l=a+1,c=t[l];if(0>Vc(o,n))l<s&&0>Vc(c,o)?(t[i]=c,t[l]=n,i=l):(t[i]=o,t[a]=n,i=a);else if(l<s&&0>Vc(c,n))t[i]=c,t[l]=n,i=l;else break e}}return e}function Vc(t,e){var n=t.sortIndex-e.sortIndex;return n!==0?n:t.id-e.id}It.unstable_now=void 0;typeof performance=="object"&&typeof performance.now=="function"?(Dv=performance,It.unstable_now=function(){return Dv.now()}):(Nd=Date,Uv=Nd.now(),It.unstable_now=function(){return Nd.now()-Uv});var Dv,Nd,Uv,ys=[],Ys=[],oE=1,pi=null,Mn=3,Fd=!1,nl=!1,il=!1,zd=!1,Nv=typeof setTimeout=="function"?setTimeout:null,Pv=typeof clearTimeout=="function"?clearTimeout:null,Iv=typeof setImmediate<"u"?setImmediate:null;function kc(t){for(var e=Zi(Ys);e!==null;){if(e.callback===null)Wc(Ys);else if(e.startTime<=t)Wc(Ys),e.sortIndex=e.expirationTime,Od(ys,e);else break;e=Zi(Ys)}}function Hd(t){if(il=!1,kc(t),!nl)if(Zi(ys)!==null)nl=!0,Ua||(Ua=!0,Da());else{var e=Zi(Ys);e!==null&&Gd(Hd,e.startTime-t)}}var Ua=!1,sl=-1,Lv=5,Ov=-1;function Fv(){return zd?!0:!(It.unstable_now()-Ov<Lv)}function Pd(){if(zd=!1,Ua){var t=It.unstable_now();Ov=t;var e=!0;try{e:{nl=!1,il&&(il=!1,Pv(sl),sl=-1),Fd=!0;var n=Mn;try{t:{for(kc(t),pi=Zi(ys);pi!==null&&!(pi.expirationTime>t&&Fv());){var i=pi.callback;if(typeof i=="function"){pi.callback=null,Mn=pi.priorityLevel;var s=i(pi.expirationTime<=t);if(t=It.unstable_now(),typeof s=="function"){pi.callback=s,kc(t),e=!0;break t}pi===Zi(ys)&&Wc(ys),kc(t)}else Wc(ys);pi=Zi(ys)}if(pi!==null)e=!0;else{var r=Zi(Ys);r!==null&&Gd(Hd,r.startTime-t),e=!1}}break e}finally{pi=null,Mn=n,Fd=!1}e=void 0}}finally{e?Da():Ua=!1}}}var Da;typeof Iv=="function"?Da=function(){Iv(Pd)}:typeof MessageChannel<"u"?(Ld=new MessageChannel,Bv=Ld.port2,Ld.port1.onmessage=Pd,Da=function(){Bv.postMessage(null)}):Da=function(){Nv(Pd,0)};var Ld,Bv;function Gd(t,e){sl=Nv(function(){t(It.unstable_now())},e)}It.unstable_IdlePriority=5;It.unstable_ImmediatePriority=1;It.unstable_LowPriority=4;It.unstable_NormalPriority=3;It.unstable_Profiling=null;It.unstable_UserBlockingPriority=2;It.unstable_cancelCallback=function(t){t.callback=null};It.unstable_forceFrameRate=function(t){0>t||125<t?console.error("forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported"):Lv=0<t?Math.floor(1e3/t):5};It.unstable_getCurrentPriorityLevel=function(){return Mn};It.unstable_next=function(t){switch(Mn){case 1:case 2:case 3:var e=3;break;default:e=Mn}var n=Mn;Mn=e;try{return t()}finally{Mn=n}};It.unstable_requestPaint=function(){zd=!0};It.unstable_runWithPriority=function(t,e){switch(t){case 1:case 2:case 3:case 4:case 5:break;default:t=3}var n=Mn;Mn=t;try{return e()}finally{Mn=n}};It.unstable_scheduleCallback=function(t,e,n){var i=It.unstable_now();switch(typeof n=="object"&&n!==null?(n=n.delay,n=typeof n=="number"&&0<n?i+n:i):n=i,t){case 1:var s=-1;break;case 2:s=250;break;case 5:s=1073741823;break;case 4:s=1e4;break;default:s=5e3}return s=n+s,t={id:oE++,callback:e,priorityLevel:t,startTime:n,expirationTime:s,sortIndex:-1},n>i?(t.sortIndex=n,Od(Ys,t),Zi(ys)===null&&t===Zi(Ys)&&(il?(Pv(sl),sl=-1):il=!0,Gd(Hd,n-i))):(t.sortIndex=s,Od(ys,t),nl||Fd||(nl=!0,Ua||(Ua=!0,Da()))),t};It.unstable_shouldYield=Fv;It.unstable_wrapCallback=function(t){var e=Mn;return function(){var n=Mn;Mn=e;try{return t.apply(this,arguments)}finally{Mn=n}}}});var Gv=Qi((Y3,Hv)=>{"use strict";Hv.exports=zv()});var Wv=Qi(En=>{"use strict";var lE=tl();function kv(t){var e="https://react.dev/errors/"+t;if(1<arguments.length){e+="?args[]="+encodeURIComponent(arguments[1]);for(var n=2;n<arguments.length;n++)e+="&args[]="+encodeURIComponent(arguments[n])}return"Minified React error #"+t+"; visit "+e+" for the full message or use the non-minified dev environment for full errors and additional helpful warnings."}function qs(){}var Bn={d:{f:qs,r:function(){throw Error(kv(522))},D:qs,C:qs,L:qs,m:qs,X:qs,S:qs,M:qs},p:0,findDOMNode:null},cE=Symbol.for("react.portal"),uE=Symbol.for("react.recoverable"),Vv=Symbol.for("react.optimistic_key");function fE(t,e,n){var i=3<arguments.length&&arguments[3]!==void 0?arguments[3]:null;return{$$typeof:cE,key:i==null?null:i===Vv?Vv:""+i,children:t,containerInfo:e,implementation:n}}var rl=lE.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;function Xc(t,e){if(t==="font")return"";if(typeof e=="string")return e==="use-credentials"?e:""}En.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE=Bn;En.browser=function(t){return{$$typeof:uE,_reason:t}};En.createPortal=function(t,e){var n=2<arguments.length&&arguments[2]!==void 0?arguments[2]:null;if(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11)throw Error(kv(299));return fE(t,e,null,n)};En.flushSync=function(t){var e=rl.T,n=Bn.p;try{if(rl.T=null,Bn.p=2,t)return t()}finally{rl.T=e,Bn.p=n,Bn.d.f()}};En.preconnect=function(t,e){typeof t=="string"&&(e?(e=e.crossOrigin,e=typeof e=="string"?e==="use-credentials"?e:"":void 0):e=null,Bn.d.C(t,e))};En.prefetchDNS=function(t){typeof t=="string"&&Bn.d.D(t)};En.preinit=function(t,e){if(typeof t=="string"&&e&&typeof e.as=="string"){var n=e.as,i=Xc(n,e.crossOrigin),s=typeof e.integrity=="string"?e.integrity:void 0,r=typeof e.fetchPriority=="string"?e.fetchPriority:void 0;n==="style"?Bn.d.S(t,typeof e.precedence=="string"?e.precedence:void 0,{crossOrigin:i,integrity:s,fetchPriority:r}):n==="script"&&Bn.d.X(t,{crossOrigin:i,integrity:s,fetchPriority:r,nonce:typeof e.nonce=="string"?e.nonce:void 0})}};En.preinitModule=function(t,e){if(typeof t=="string")if(typeof e=="object"&&e!==null){if(e.as==null||e.as==="script"){var n=Xc(e.as,e.crossOrigin);Bn.d.M(t,{crossOrigin:n,integrity:typeof e.integrity=="string"?e.integrity:void 0,nonce:typeof e.nonce=="string"?e.nonce:void 0,fetchPriority:typeof e.fetchPriority=="string"?e.fetchPriority:void 0})}}else e==null&&Bn.d.M(t)};En.preload=function(t,e){if(typeof t=="string"&&typeof e=="object"&&e!==null&&typeof e.as=="string"){var n=e.as,i=Xc(n,e.crossOrigin);Bn.d.L(t,n,{crossOrigin:i,integrity:typeof e.integrity=="string"?e.integrity:void 0,nonce:typeof e.nonce=="string"?e.nonce:void 0,type:typeof e.type=="string"?e.type:void 0,fetchPriority:typeof e.fetchPriority=="string"?e.fetchPriority:void 0,referrerPolicy:typeof e.referrerPolicy=="string"?e.referrerPolicy:void 0,imageSrcSet:typeof e.imageSrcSet=="string"?e.imageSrcSet:void 0,imageSizes:typeof e.imageSizes=="string"?e.imageSizes:void 0,media:typeof e.media=="string"?e.media:void 0})}};En.preloadModule=function(t,e){if(typeof t=="string")if(e){var n=Xc(e.as,e.crossOrigin);Bn.d.m(t,{as:typeof e.as=="string"&&e.as!=="script"?e.as:void 0,crossOrigin:n,integrity:typeof e.integrity=="string"?e.integrity:void 0,nonce:typeof e.nonce=="string"?e.nonce:void 0,fetchPriority:typeof e.fetchPriority=="string"?e.fetchPriority:void 0})}else Bn.d.m(t)};En.requestFormReset=function(t){Bn.d.r(t)};En.unstable_batchedUpdates=function(t,e){return t(e)};En.useFormState=function(t,e,n){return rl.H.useFormState(t,e,n)};En.useFormStatus=function(){return rl.H.useHostTransitionStatus()};En.version="19.3.0"});var qv=Qi((Q3,Yv)=>{"use strict";function Xv(){if(!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__>"u"||typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE!="function"))try{__REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(Xv)}catch(t){console.error(t)}}Xv(),Yv.exports=Wv()});var BA=Qi(wf=>{"use strict";var sn=Gv(),Iy=tl(),hE=qv();function J(t){var e="https://react.dev/errors/"+t;if(1<arguments.length){e+="?args[]="+encodeURIComponent(arguments[1]);for(var n=2;n<arguments.length;n++)e+="&args[]="+encodeURIComponent(arguments[n])}return"Minified React error #"+t+"; visit "+e+" for the full message or use the non-minified dev environment for full errors and additional helpful warnings."}function By(t){return!(!t||t.nodeType!==1&&t.nodeType!==9&&t.nodeType!==11)}function Yl(t){for(var e=t,n=e;n&&!n.alternate;)e=n,(e.flags&4098)!==0&&(t=e.return),n=e.return;for(;e.return;)e=e.return;return e.tag===3?t:null}function Ny(t){if(t.tag===13){var e=t.memoizedState;if(e===null&&(t=t.alternate,t!==null&&(e=t.memoizedState)),e!==null)return e.dehydrated}return null}function Py(t){if(t.tag===31){var e=t.memoizedState;if(e===null&&(t=t.alternate,t!==null&&(e=t.memoizedState)),e!==null)return e.dehydrated}return null}function Qv(t){if(Yl(t)!==t)throw Error(J(188))}function dE(t){var e=t.alternate;if(!e){if(e=Yl(t),e===null)throw Error(J(188));return e!==t?null:t}for(var n=t,i=e;;){var s=n.return;if(s===null)break;var r=s.alternate;if(r===null){if(i=s.return,i!==null){n=i;continue}break}if(s.child===r.child){for(r=s.child;r;){if(r===n)return Qv(s),t;if(r===i)return Qv(s),e;r=r.sibling}throw Error(J(188))}if(n.return!==i.return)n=s,i=r;else{for(var a=!1,o=s.child;o;){if(o===n){a=!0,n=s,i=r;break}if(o===i){a=!0,i=s,n=r;break}o=o.sibling}if(!a){for(o=r.child;o;){if(o===n){a=!0,n=r,i=s;break}if(o===i){a=!0,i=r,n=s;break}o=o.sibling}if(!a)throw Error(J(189))}}if(n.alternate!==i)throw Error(J(190))}if(n.tag!==3)throw Error(J(188));return n.stateNode.current===n?t:e}function Ly(t){var e=t.tag;if(e===5||e===26||e===27||e===6)return t;for(t=t.child;t!==null;){if(e=Ly(t),e!==null)return e;t=t.sibling}return null}function Yn(t,e,n,i,s,r){for(;t!==null;){if((t.tag===5||t.tag===27||t.tag===6)&&n(t,i,s,r)||(t.tag!==22||t.memoizedState===null)&&(e||t.tag!==5&&t.tag!==27)&&Yn(t.child,e,n,i,s,r))return!0;t=t.sibling}return!1}function la(t){for(t=t.return;t!==null;){if(t.tag===3||t.tag===5||t.tag===27)return t;t=t.return}return null}function Zv(t){var e=!1;for(t=t.return;t!==null&&(t.tag===4&&(e=!0),!(t.tag===3||t.tag===5||t.tag===27));)t=t.return;return e}function Oy(t){var e=[null,null],n=la(t);return n===null||Fy(e,t,n.child,{foundSelf:!1}),e}function Fy(t,e,n,i){for(;n!==null;){if(n===e)i.foundSelf=!0;else if(n.tag===5||n.tag===27||n.tag===6){if(i.foundSelf)return t[1]=n,!0;t[0]=n}else if((n.tag!==22||n.memoizedState===null)&&Fy(t,e,n.child,i))return!0;n=n.sibling}return!1}function nn(t){switch(t.tag){case 5:case 27:case 6:return t.stateNode;case 3:return t.stateNode.containerInfo;default:throw Error(J(559))}}var Fa=null,yp=null;function pE(t,e,n){return t===n?!0:t===e?(Fa=t,!0):!1}function mE(t,e,n){return t===n?(yp=t,!1):t===e?(yp!==null&&(Fa=t),!0):!1}function Kv(t){if(t===null)return null;do t=t===null?null:t.return;while(t&&t.tag!==5&&t.tag!==27&&t.tag!==3);return t||null}function _p(t,e,n){for(var i=0,s=t;s;s=n(s))i++;s=0;for(var r=e;r;r=n(r))s++;for(;0<i-s;)t=n(t),i--;for(;0<s-i;)e=n(e),s--;for(;i--;){if(t===e||e!==null&&t===e.alternate)return t;t=n(t),e=n(e)}return null}var yt=Object.assign,gE=Symbol.for("react.element"),Yc=Symbol.for("react.transitional.element"),hl=Symbol.for("react.portal"),za=Symbol.for("react.fragment"),zy=Symbol.for("react.strict_mode"),Sp=Symbol.for("react.profiler"),Hy=Symbol.for("react.consumer"),ts=Symbol.for("react.context"),Dm=Symbol.for("react.forward_ref"),Ap=Symbol.for("react.suspense"),Mp=Symbol.for("react.suspense_list"),Um=Symbol.for("react.memo"),Js=Symbol.for("react.lazy"),Ep=Symbol.for("react.activity"),vE=Symbol.for("react.legacy_hidden"),xE=Symbol.for("react.memo_cache_sentinel"),Tp=Symbol.for("react.view_transition"),yE=Symbol.for("react.recoverable"),Jv=Symbol.iterator;function al(t){return t===null||typeof t!="object"?null:(t=Jv&&t[Jv]||t["@@iterator"],typeof t=="function"?t:null)}var _E=Symbol.for("react.client.reference");function bp(t){if(t==null)return null;if(typeof t=="function")return t.$$typeof===_E?null:t.displayName||t.name||null;if(typeof t=="string")return t;switch(t){case za:return"Fragment";case Sp:return"Profiler";case zy:return"StrictMode";case Ap:return"Suspense";case Mp:return"SuspenseList";case Ep:return"Activity";case Tp:return"ViewTransition"}if(typeof t=="object")switch(t.$$typeof){case hl:return"Portal";case ts:return t.displayName||"Context";case Hy:return(t._context.displayName||"Context")+".Consumer";case Dm:var e=t.render;return t=t.displayName,t||(t=e.displayName||e.name||"",t=t!==""?"ForwardRef("+t+")":"ForwardRef"),t;case Um:return e=t.displayName||null,e!==null?e:bp(t.type)||"Memo";case Js:e=t._payload,t=t._init;try{return bp(t(e))}catch{}}return null}var dl=Array.isArray,Ue=Iy.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,rt=hE.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,Zr={pending:!1,data:null,method:null,action:null},wp=[],Ha=-1;function ls(t){return{current:t}}function vn(t){0>Ha||(t.current=wp[Ha],wp[Ha]=null,Ha--)}function bt(t,e){Ha++,wp[Ha]=t.current,t.current=e}var rs=ls(null),Dl=ls(null),ar=ls(null),Iu=ls(null);function Bu(t,e){switch(bt(ar,e),bt(Dl,t),bt(rs,null),e.nodeType){case 9:case 11:t=(t=e.documentElement)&&(t=t.namespaceURI)?fy(t):0;break;default:if(t=e.tagName,e=e.namespaceURI)e=fy(e),t=lA(e,t);else switch(t){case"svg":t=1;break;case"math":t=2;break;default:t=0}}vn(rs),bt(rs,t)}function ao(){vn(rs),vn(Dl),vn(ar)}function Cp(t){var e=t.memoizedState;e!==null&&(vo._currentValue=e.memoizedState,bt(Iu,t)),e=rs.current;var n=lA(e,t.type);e!==n&&(bt(Dl,t),bt(rs,n))}function Nu(t){Dl.current===t&&(vn(rs),vn(Dl)),Iu.current===t&&(vn(Iu),vo._currentValue=Zr)}var Vd,jv;function Zs(t){if(Vd===void 0)try{throw Error()}catch(n){var e=n.stack.trim().match(/\n( *(at )?)/);Vd=e&&e[1]||"",jv=-1<n.stack.indexOf(`
    at`)?" (<anonymous>)":-1<n.stack.indexOf("@")?"@unknown:0:0":""}return`
`+Vd+t+jv}var kd=!1;function Wd(t,e){if(!t||kd)return"";kd=!0;var n=Error.prepareStackTrace;Error.prepareStackTrace=void 0;try{var i={DetermineComponentFrameRoot:function(){try{if(e){var p=function(){throw Error()};if(Object.defineProperty(p.prototype,"props",{set:function(){throw Error()}}),typeof Reflect=="object"&&Reflect.construct){try{Reflect.construct(p,[])}catch(v){var u=v}Reflect.construct(t,[],p)}else{try{p.call()}catch(v){u=v}p=!1;try{var d=Object.getOwnPropertyDescriptor(t.prototype,"props");Object.defineProperty(t.prototype,"props",{configurable:!0,set:function(){throw Error()}}),p=!0,new t}finally{p&&(d!==void 0?Object.defineProperty(t.prototype,"props",d):delete t.prototype.props)}}}else{try{throw Error()}catch(v){u=v}(p=t())&&typeof p.catch=="function"&&p.catch(function(){})}}catch(v){if(v&&u&&typeof v.stack=="string")return[v.stack,u.stack]}return[null,null]}};i.DetermineComponentFrameRoot.displayName="DetermineComponentFrameRoot";var s=Object.getOwnPropertyDescriptor(i.DetermineComponentFrameRoot,"name");s&&s.configurable&&Object.defineProperty(i.DetermineComponentFrameRoot,"name",{value:"DetermineComponentFrameRoot"});var r=i.DetermineComponentFrameRoot(),a=r[0],o=r[1];if(a&&o){var l=a.split(`
`),c=o.split(`
`);for(s=i=0;i<l.length&&!l[i].includes("DetermineComponentFrameRoot");)i++;for(;s<c.length&&!c[s].includes("DetermineComponentFrameRoot");)s++;if(i===l.length||s===c.length)for(i=l.length-1,s=c.length-1;1<=i&&0<=s&&l[i]!==c[s];)s--;for(;1<=i&&0<=s;i--,s--)if(l[i]!==c[s]){if(i!==1||s!==1)do if(i--,s--,0>s||l[i]!==c[s]){var h=`
`+l[i].replace(" at new "," at ");return t.displayName&&h.includes("<anonymous>")&&(h=h.replace("<anonymous>",t.displayName)),h}while(1<=i&&0<=s);break}}}finally{kd=!1,Error.prepareStackTrace=n}return(n=t?t.displayName||t.name:"")?Zs(n):""}function SE(t,e){switch(t.tag){case 26:case 27:case 5:return Zs(t.type);case 16:return Zs("Lazy");case 13:return t.child!==e&&e!==null?Zs("Suspense Fallback"):Zs("Suspense");case 19:return Zs("SuspenseList");case 0:case 15:return Wd(t.type,!1);case 11:return Wd(t.type.render,!1);case 1:return Wd(t.type,!0);case 31:return Zs("Activity");case 30:return Zs("ViewTransition");default:return""}}function $v(t){try{var e="",n=null;do e+=SE(t,n),n=t,t=t.return;while(t);return e}catch(i){return`
Error generating stack: `+i.message+`
`+i.stack}}var Rp=Object.prototype.hasOwnProperty,Im=sn.unstable_scheduleCallback,Xd=sn.unstable_cancelCallback,AE=sn.unstable_shouldYield,ME=sn.unstable_requestPaint,ni=sn.unstable_now,EE=sn.unstable_getCurrentPriorityLevel,Gy=sn.unstable_ImmediatePriority,Vy=sn.unstable_UserBlockingPriority,Pu=sn.unstable_NormalPriority,TE=sn.unstable_LowPriority,ky=sn.unstable_IdlePriority,bE=sn.log,wE=sn.unstable_setDisableYieldValue,ql=null,ii=null;function er(t){if(typeof bE=="function"&&wE(t),ii&&typeof ii.setStrictMode=="function")try{ii.setStrictMode(ql,t)}catch{}}var si=Math.clz32?Math.clz32:DE,CE=Math.log,RE=Math.LN2;function DE(t){return t>>>=0,t===0?32:31-(CE(t)/RE|0)|0}var qc=256,Qc=262144,Zc=4194304;function Wr(t){var e=t&42;if(e!==0)return e;switch(t&-t){case 1:return 1;case 2:return 2;case 4:return 4;case 8:return 8;case 16:return 16;case 32:return 32;case 64:return 64;case 128:return 128;case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:return t&-t;case 262144:case 524288:case 1048576:case 2097152:return t&3932160;case 4194304:case 8388608:case 16777216:case 33554432:return t&62914560;case 67108864:return 67108864;case 134217728:return 134217728;case 268435456:return 268435456;case 536870912:return 536870912;case 1073741824:return 0;default:return t}}function lf(t,e,n){var i=t.pendingLanes;if(i===0)return 0;var s=0,r=t.suspendedLanes,a=t.pingedLanes;t=t.warmLanes;var o=i&134217727;return o!==0?(i=o&~r,i!==0?s=Wr(i):(a&=o,a!==0?s=Wr(a):n||(n=o&~t,n!==0&&(s=Wr(n))))):(o=i&~r,o!==0?s=Wr(o):a!==0?s=Wr(a):n||(n=i&~t,n!==0&&(s=Wr(n)))),s===0?0:e!==0&&e!==s&&(e&r)===0&&(r=s&-s,n=e&-e,r>=n||r===32&&(n&4194048)!==0)?e:s}function Ql(t,e){return(t.pendingLanes&~(t.suspendedLanes&~t.pingedLanes)&e)===0}function Wy(t,e){(e&8)!==0&&(e|=e&32);var n=t.entangledLanes;if(n!==0)for(t=t.entanglements,n&=e;0<n;){var i=31-si(n),s=1<<i;e|=t[i],n&=~s}return e}function UE(t,e){switch(t){case 1:case 2:case 4:case 8:case 64:return e+250;case 16:case 32:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return e+5e3;case 4194304:case 8388608:case 16777216:case 33554432:return-1;case 67108864:case 134217728:case 268435456:case 536870912:case 1073741824:return-1;default:return-1}}function Xy(){var t=Zc;return Zc<<=1,(Zc&62914560)===0&&(Zc=4194304),t}function Yd(t){for(var e=[],n=0;31>n;n++)e.push(t);return e}function Zl(t,e){t.pendingLanes|=e,e!==268435456&&(t.suspendedLanes=0,t.pingedLanes=0,t.warmLanes=0)}function IE(t,e,n,i,s,r){var a=t.pendingLanes;t.pendingLanes=n,t.suspendedLanes=0,t.pingedLanes=0,t.warmLanes=0,t.expiredLanes&=n,t.entangledLanes&=n,t.errorRecoveryDisabledLanes&=n,t.shellSuspendCounter=0;var o=t.entanglements,l=t.expirationTimes,c=t.hiddenUpdates;for(n=a&~n;0<n;){var h=31-si(n),p=1<<h;o[h]=0,l[h]=-1;var u=c[h];if(u!==null)for(c[h]=null,h=0;h<u.length;h++){var d=u[h];d!==null&&(d.lane&=-536870913)}n&=~p}i!==0&&Yy(t,i,0),r!==0&&s===0&&t.tag!==0&&(t.suspendedLanes|=r&~(a&~e))}function Yy(t,e,n){t.pendingLanes|=e,t.suspendedLanes&=~e;var i=31-si(e);t.entangledLanes|=e,t.entanglements[i]=t.entanglements[i]|1073741824|n&261930}function qy(t,e){var n=t.entangledLanes|=e;for(t=t.entanglements;n;){var i=31-si(n),s=1<<i;s&e|t[i]&e&&(t[i]|=e),n&=~s}}function Qy(t,e){var n=e&-e;return n=(n&42)!==0?1:Bm(n),(n&(t.suspendedLanes|e))!==0?0:n}function Bm(t){switch(t){case 2:t=1;break;case 8:t=4;break;case 32:t=16;break;case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:case 4194304:case 8388608:case 16777216:case 33554432:t=128;break;case 268435456:t=134217728;break;default:t=0}return t}function Nm(t){return t&=-t,2<t?8<t?(t&134217727)!==0?32:268435456:8:2}function Zy(){var t=rt.p;return t!==0?t:(t=window.event,t===void 0?32:DA(t.type))}function ex(t,e){var n=rt.p;try{return rt.p=t,e()}finally{rt.p=n}}var Is=Math.random().toString(36).slice(2),mn="__reactFiber$"+Is,qn="__reactProps$"+Is,_o="__reactContainer$"+Is,tx="__reactEvents$"+Is,BE="__reactListeners$"+Is,NE="__reactHandles$"+Is,nx="__reactResources$"+Is,Kl="__reactMarker$"+Is,Lu="__reactLoad$"+Is;function cf(t){delete t[mn],delete t[qn],delete t[BE],delete t[NE]}function qr(t){var e;if(e=t[mn])return e;for(var n=t.parentNode;n;){if(e=n[_o]||n[mn]){if(n=e.alternate,e.child!==null||n!==null&&n.child!==null)for(t=yy(t);t!==null;){if(n=t[mn])return n;t=yy(t)}return e}t=n,n=t.parentNode}return null}function So(t){if(t=t[mn]||t[_o]){var e=t.tag;if(e===5||e===6||e===13||e===31||e===26||e===27||e===3)return t}return null}function pl(t){var e=t.tag;if(e===5||e===26||e===27||e===6)return t.stateNode;throw Error(J(33))}function Ka(t){var e=t[nx];return e||(e=t[nx]={hoistableStyles:new Map,hoistableScripts:new Map}),e}function un(t){t[Kl]=!0}function Ky(t){t[Lu]=void 0}var Jy=new Set,jy={};function ca(t,e){oo(t,e),oo(t+"Capture",e)}function oo(t,e){for(jy[t]=e,t=0;t<e.length;t++)Jy.add(e[t])}var PE=RegExp("^[:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD][:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$"),ix={},sx={};function LE(t){return Rp.call(sx,t)?!0:Rp.call(ix,t)?!1:PE.test(t)?sx[t]=!0:(ix[t]=!0,!1)}var nt=!1;function rx(){var t=nt;return nt=!1,t}function du(t,e,n){if(LE(e))if(n===null)t.removeAttribute(e);else{switch(typeof n){case"undefined":case"function":case"symbol":t.removeAttribute(e);return;case"boolean":var i=e.toLowerCase().slice(0,5);if(i!=="data-"&&i!=="aria-"){t.removeAttribute(e);return}}t.setAttribute(e,n)}}function Kc(t,e,n){if(n===null)t.removeAttribute(e);else{switch(typeof n){case"undefined":case"function":case"symbol":case"boolean":t.removeAttribute(e);return}t.setAttribute(e,n)}}function _s(t,e,n,i){if(i===null)t.removeAttribute(n);else{switch(typeof i){case"undefined":case"function":case"symbol":case"boolean":t.removeAttribute(n);return}t.setAttributeNS(e,n,i)}}function jn(t){switch(typeof t){case"bigint":case"boolean":case"number":case"string":case"undefined":return t;case"object":return t;default:return""}}function $y(t){var e=t.type;return(t=t.nodeName)&&t.toLowerCase()==="input"&&(e==="checkbox"||e==="radio")}function OE(t,e,n){var i=Object.getOwnPropertyDescriptor(t.constructor.prototype,e);if(!t.hasOwnProperty(e)&&typeof i<"u"&&typeof i.get=="function"&&typeof i.set=="function"){var s=i.get,r=i.set;return Object.defineProperty(t,e,{configurable:!0,get:function(){return s.call(this)},set:function(a){n=""+a,r.call(this,a)}}),Object.defineProperty(t,e,{enumerable:i.enumerable}),{getValue:function(){return n},setValue:function(a){n=""+a},stopTracking:function(){t._valueTracker=null,delete t[e]}}}}function Dp(t){if(!t._valueTracker){var e=$y(t)?"checked":"value";t._valueTracker=OE(t,e,""+t[e])}}function e_(t){if(!t)return!1;var e=t._valueTracker;if(!e)return!0;var n=e.getValue(),i="";return t&&(i=$y(t)?t.checked?"true":"false":t.value),t=i,t!==n?(e.setValue(t),!0):!1}var FE=/[\n"\\]/g;function yi(t){return t.replace(FE,function(e){return"\\"+e.charCodeAt(0).toString(16)+" "})}function Up(t,e,n,i,s,r,a,o){t.name="",a!=null&&typeof a!="function"&&typeof a!="symbol"&&typeof a!="boolean"?t.type=a:t.removeAttribute("type"),e!=null?a==="number"?(e===0&&t.value===""||t.value!=e)&&(t.value=""+jn(e)):t.value!==""+jn(e)&&(t.value=""+jn(e)):a!=="submit"&&a!=="reset"||t.removeAttribute("value"),e!=null?a==="number"&&t.value==e?qd(t,jn(t.value)):qd(t,jn(e)):n!=null?qd(t,jn(n)):i!=null&&t.removeAttribute("value"),s==null&&r!=null&&(t.defaultChecked=!!r),s!=null&&(t.checked=s&&typeof s!="function"&&typeof s!="symbol"),o!=null&&typeof o!="function"&&typeof o!="symbol"&&typeof o!="boolean"?t.name=""+jn(o):t.removeAttribute("name")}function t_(t,e,n,i,s,r,a,o){if(r!=null&&typeof r!="function"&&typeof r!="symbol"&&typeof r!="boolean"&&(t.type=r),e!=null||n!=null){if(!(r!=="submit"&&r!=="reset"||e!=null)){Dp(t);return}n=n!=null?""+jn(n):"",e=e!=null?""+jn(e):n,o||e===t.value||(t.value=e),t.defaultValue=e}i=i??s,i=typeof i!="function"&&typeof i!="symbol"&&!!i,t.checked=o?t.checked:!!i,t.defaultChecked=!!i,a!=null&&typeof a!="function"&&typeof a!="symbol"&&typeof a!="boolean"&&(t.name=a),Dp(t)}function qd(t,e){t.defaultValue!==""+e&&(t.defaultValue=""+e)}function Ja(t,e,n,i){if(t=t.options,e){e={};for(var s=0;s<n.length;s++)e["$"+n[s]]=!0;for(n=0;n<t.length;n++)s=e.hasOwnProperty("$"+t[n].value),t[n].selected!==s&&(t[n].selected=s),s&&i&&(t[n].defaultSelected=!0)}else{for(n=""+jn(n),e=null,s=0;s<t.length;s++){if(t[s].value===n){t[s].selected=!0,i&&(t[s].defaultSelected=!0);return}e!==null||t[s].disabled||(e=t[s])}e!==null&&(e.selected=!0)}}function n_(t,e,n){if(e!=null&&(e=""+jn(e),e!==t.value&&(t.value=e),n==null)){t.defaultValue!==e&&(t.defaultValue=e);return}t.defaultValue=n!=null?""+jn(n):""}function i_(t,e,n,i){if(e==null){if(i!=null){if(n!=null)throw Error(J(92));if(dl(i)){if(1<i.length)throw Error(J(93));i=i[0]}n=i}n==null&&(n=""),e=n}n=jn(e),t.defaultValue=n,i=t.textContent,i===n&&i!==""&&i!==null&&(t.value=i),Dp(t)}function lo(t,e){if(e){var n=t.firstChild;if(n&&n===t.lastChild&&n.nodeType===3){n.nodeValue=e;return}}t.textContent=e}var zE=new Set("animationIterationCount aspectRatio borderImageOutset borderImageSlice borderImageWidth boxFlex boxFlexGroup boxOrdinalGroup columnCount columns flex flexGrow flexPositive flexShrink flexNegative flexOrder gridArea gridRow gridRowEnd gridRowSpan gridRowStart gridColumn gridColumnEnd gridColumnSpan gridColumnStart fontWeight lineClamp lineHeight opacity order orphans scale tabSize widows zIndex zoom fillOpacity floodOpacity stopOpacity strokeDasharray strokeDashoffset strokeMiterlimit strokeOpacity strokeWidth MozAnimationIterationCount MozBoxFlex MozBoxFlexGroup MozLineClamp msAnimationIterationCount msFlex msZoom msFlexGrow msFlexNegative msFlexOrder msFlexPositive msFlexShrink msGridColumn msGridColumnSpan msGridRow msGridRowSpan WebkitAnimationIterationCount WebkitBoxFlex WebKitBoxFlexGroup WebkitBoxOrdinalGroup WebkitColumnCount WebkitColumns WebkitFlex WebkitFlexGrow WebkitFlexPositive WebkitFlexShrink WebkitLineClamp".split(" "));function ax(t,e,n){var i=e.indexOf("--")===0;n==null||typeof n=="boolean"||n===""?i?t.setProperty(e,""):e==="float"?t.cssFloat="":t[e]="":i?t.setProperty(e,n):typeof n!="number"||n===0||zE.has(e)?e==="float"?t.cssFloat=n:t[e]=(""+n).trim():t[e]=n+"px"}function s_(t,e,n){if(e!=null&&typeof e!="object")throw Error(J(62));if(t=t.style,n!=null){for(var i in n)!n.hasOwnProperty(i)||e!=null&&e.hasOwnProperty(i)||(i.indexOf("--")===0?t.setProperty(i,""):i==="float"?t.cssFloat="":t[i]="",nt=!0);for(var s in e)i=e[s],e.hasOwnProperty(s)&&n[s]!==i&&(ax(t,s,i),nt=!0)}else for(var r in e)e.hasOwnProperty(r)&&ax(t,r,e[r])}function Pm(t){if(t.indexOf("-")===-1)return!1;switch(t){case"annotation-xml":case"color-profile":case"font-face":case"font-face-src":case"font-face-uri":case"font-face-format":case"font-face-name":case"missing-glyph":return!1;default:return!0}}var HE=new Map([["acceptCharset","accept-charset"],["htmlFor","for"],["httpEquiv","http-equiv"],["crossOrigin","crossorigin"],["accentHeight","accent-height"],["alignmentBaseline","alignment-baseline"],["arabicForm","arabic-form"],["baselineShift","baseline-shift"],["capHeight","cap-height"],["clipPath","clip-path"],["clipRule","clip-rule"],["colorInterpolation","color-interpolation"],["colorInterpolationFilters","color-interpolation-filters"],["colorProfile","color-profile"],["colorRendering","color-rendering"],["dominantBaseline","dominant-baseline"],["enableBackground","enable-background"],["fillOpacity","fill-opacity"],["fillRule","fill-rule"],["floodColor","flood-color"],["floodOpacity","flood-opacity"],["fontFamily","font-family"],["fontSize","font-size"],["fontSizeAdjust","font-size-adjust"],["fontStretch","font-stretch"],["fontStyle","font-style"],["fontVariant","font-variant"],["fontWeight","font-weight"],["glyphName","glyph-name"],["glyphOrientationHorizontal","glyph-orientation-horizontal"],["glyphOrientationVertical","glyph-orientation-vertical"],["horizAdvX","horiz-adv-x"],["horizOriginX","horiz-origin-x"],["imageRendering","image-rendering"],["letterSpacing","letter-spacing"],["lightingColor","lighting-color"],["markerEnd","marker-end"],["markerMid","marker-mid"],["markerStart","marker-start"],["maskType","mask-type"],["overlinePosition","overline-position"],["overlineThickness","overline-thickness"],["paintOrder","paint-order"],["panose-1","panose-1"],["pointerEvents","pointer-events"],["renderingIntent","rendering-intent"],["shapeRendering","shape-rendering"],["stopColor","stop-color"],["stopOpacity","stop-opacity"],["strikethroughPosition","strikethrough-position"],["strikethroughThickness","strikethrough-thickness"],["strokeDasharray","stroke-dasharray"],["strokeDashoffset","stroke-dashoffset"],["strokeLinecap","stroke-linecap"],["strokeLinejoin","stroke-linejoin"],["strokeMiterlimit","stroke-miterlimit"],["strokeOpacity","stroke-opacity"],["strokeWidth","stroke-width"],["textAnchor","text-anchor"],["textDecoration","text-decoration"],["textRendering","text-rendering"],["transformOrigin","transform-origin"],["underlinePosition","underline-position"],["underlineThickness","underline-thickness"],["unicodeBidi","unicode-bidi"],["unicodeRange","unicode-range"],["unitsPerEm","units-per-em"],["vAlphabetic","v-alphabetic"],["vHanging","v-hanging"],["vIdeographic","v-ideographic"],["vMathematical","v-mathematical"],["vectorEffect","vector-effect"],["vertAdvY","vert-adv-y"],["vertOriginX","vert-origin-x"],["vertOriginY","vert-origin-y"],["wordSpacing","word-spacing"],["writingMode","writing-mode"],["xmlnsXlink","xmlns:xlink"],["xHeight","x-height"]]),GE=/^[\u0000-\u001F ]*j[\r\n\t]*a[\r\n\t]*v[\r\n\t]*a[\r\n\t]*s[\r\n\t]*c[\r\n\t]*r[\r\n\t]*i[\r\n\t]*p[\r\n\t]*t[\r\n\t]*:/i;function pu(t){return GE.test(""+t)?"javascript:throw new Error('React has blocked a javascript: URL as a security precaution.')":t}function ns(){}var Ip=null;function Lm(t){return t=t.target||t.srcElement||window,t.correspondingUseElement&&(t=t.correspondingUseElement),t.nodeType===3?t.parentNode:t}var Ga=null,ja=null;function ox(t){var e=So(t);if(e&&(t=e.stateNode)){var n=t[qn]||null;e:switch(t=e.stateNode,e.type){case"input":if(Up(t,n.value,n.defaultValue,n.defaultValue,n.checked,n.defaultChecked,n.type,n.name),e=n.name,n.type==="radio"&&e!=null){for(n=t;n.parentNode;)n=n.parentNode;for(n=n.querySelectorAll('input[name="'+yi(""+e)+'"][type="radio"]'),e=0;e<n.length;e++){var i=n[e];if(i!==t&&i.form===t.form){var s=i[qn]||null;if(!s)throw Error(J(90));Up(i,s.value,s.defaultValue,s.defaultValue,s.checked,s.defaultChecked,s.type,s.name)}}for(e=0;e<n.length;e++)i=n[e],i.form===t.form&&e_(i)}break e;case"textarea":n_(t,n.value,n.defaultValue);break e;case"select":e=n.value,e!=null&&Ja(t,!!n.multiple,e,!1)}}}var Qd=!1;function r_(t,e,n){if(Qd)return t(e,n);Qd=!0;try{var i=t(e);return i}finally{if(Qd=!1,(Ga!==null||ja!==null)&&(Mf(),Ga&&(e=Ga,t=ja,ja=Ga=null,ox(e),t)))for(e=0;e<t.length;e++)ox(t[e])}}function Ul(t,e){var n=t.stateNode;if(n===null)return null;var i=n[qn]||null;if(i===null)return null;n=i[e];e:switch(e){case"onClick":case"onClickCapture":case"onDoubleClick":case"onDoubleClickCapture":case"onMouseDown":case"onMouseDownCapture":case"onMouseMove":case"onMouseMoveCapture":case"onMouseUp":case"onMouseUpCapture":case"onMouseEnter":(i=!i.disabled)||(t=t.type,i=!(t==="button"||t==="input"||t==="select"||t==="textarea")),t=!i;break e;default:t=!1}if(t)return null;if(n&&typeof n!="function")throw Error(J(231,e,typeof n));return n}var bs=!(typeof window>"u"||typeof window.document>"u"||typeof window.document.createElement>"u"),Bp=!1;if(bs)try{Ia={},Object.defineProperty(Ia,"passive",{get:function(){Bp=!0}}),window.addEventListener("test",Ia,Ia),window.removeEventListener("test",Ia,Ia)}catch{Bp=!1}var Ia,tr=null,Om=null,mu=null;function a_(){if(mu)return mu;var t,e=Om,n=e.length,i,s="value"in tr?tr.value:tr.textContent,r=s.length;for(t=0;t<n&&e[t]===s[t];t++);var a=n-t;for(i=1;i<=a&&e[n-i]===s[r-i];i++);return mu=s.slice(t,1<i?1-i:void 0)}function gu(t){var e=t.keyCode;return"charCode"in t?(t=t.charCode,t===0&&e===13&&(t=13)):t=e,t===10&&(t=13),32<=t||t===13?t:0}function Jc(){return!0}function lx(){return!1}function On(t){function e(n,i,s,r,a){this._reactName=n,this._targetInst=s,this.type=i,this.nativeEvent=r,this.target=a,this.currentTarget=null;for(var o in t)t.hasOwnProperty(o)&&(n=t[o],this[o]=n?n(r):r[o]);return this.isDefaultPrevented=(r.defaultPrevented!=null?r.defaultPrevented:r.returnValue===!1)?Jc:lx,this.isPropagationStopped=lx,this}return yt(e.prototype,{preventDefault:function(){this.defaultPrevented=!0;var n=this.nativeEvent;n&&(n.preventDefault?n.preventDefault():typeof n.returnValue!="unknown"&&(n.returnValue=!1),this.isDefaultPrevented=Jc)},stopPropagation:function(){var n=this.nativeEvent;n&&(n.stopPropagation?n.stopPropagation():typeof n.cancelBubble!="unknown"&&(n.cancelBubble=!0),this.isPropagationStopped=Jc)},persist:function(){},isPersistent:Jc}),e}var Sr={eventPhase:0,bubbles:0,cancelable:0,timeStamp:function(t){return t.timeStamp||Date.now()},defaultPrevented:0,isTrusted:0},uf=On(Sr),Jl=yt({},Sr,{view:0,detail:0}),VE=On(Jl),Zd,Kd,ol,ff=yt({},Jl,{screenX:0,screenY:0,clientX:0,clientY:0,pageX:0,pageY:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,getModifierState:Fm,button:0,buttons:0,relatedTarget:function(t){return t.relatedTarget===void 0?t.fromElement===t.srcElement?t.toElement:t.fromElement:t.relatedTarget},movementX:function(t){return"movementX"in t?t.movementX:(t!==ol&&(ol&&t.type==="mousemove"?(Zd=t.screenX-ol.screenX,Kd=t.screenY-ol.screenY):Kd=Zd=0,ol=t),Zd)},movementY:function(t){return"movementY"in t?t.movementY:Kd}}),cx=On(ff),kE=yt({},ff,{dataTransfer:0}),WE=On(kE),XE=yt({},Jl,{relatedTarget:0}),Jd=On(XE),YE=yt({},Sr,{animationName:0,elapsedTime:0,pseudoElement:0}),qE=On(YE),QE=yt({},Sr,{clipboardData:function(t){return"clipboardData"in t?t.clipboardData:window.clipboardData}}),ZE=On(QE),KE=yt({},Sr,{data:0}),ux=On(KE),JE={Esc:"Escape",Spacebar:" ",Left:"ArrowLeft",Up:"ArrowUp",Right:"ArrowRight",Down:"ArrowDown",Del:"Delete",Win:"OS",Menu:"ContextMenu",Apps:"ContextMenu",Scroll:"ScrollLock",MozPrintableKey:"Unidentified"},jE={8:"Backspace",9:"Tab",12:"Clear",13:"Enter",16:"Shift",17:"Control",18:"Alt",19:"Pause",20:"CapsLock",27:"Escape",32:" ",33:"PageUp",34:"PageDown",35:"End",36:"Home",37:"ArrowLeft",38:"ArrowUp",39:"ArrowRight",40:"ArrowDown",45:"Insert",46:"Delete",112:"F1",113:"F2",114:"F3",115:"F4",116:"F5",117:"F6",118:"F7",119:"F8",120:"F9",121:"F10",122:"F11",123:"F12",144:"NumLock",145:"ScrollLock",224:"Meta"},$E={Alt:"altKey",Control:"ctrlKey",Meta:"metaKey",Shift:"shiftKey"};function eT(t){var e=this.nativeEvent;return e.getModifierState?e.getModifierState(t):(t=$E[t])?!!e[t]:!1}function Fm(){return eT}var tT=yt({},Jl,{key:function(t){if(t.key){var e=JE[t.key]||t.key;if(e!=="Unidentified")return e}return t.type==="keypress"?(t=gu(t),t===13?"Enter":String.fromCharCode(t)):t.type==="keydown"||t.type==="keyup"?jE[t.keyCode]||"Unidentified":""},code:0,location:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,repeat:0,locale:0,getModifierState:Fm,charCode:function(t){return t.type==="keypress"?gu(t):0},keyCode:function(t){return t.type==="keydown"||t.type==="keyup"?t.keyCode:0},which:function(t){return t.type==="keypress"?gu(t):t.type==="keydown"||t.type==="keyup"?t.keyCode:0}}),nT=On(tT),iT=yt({},ff,{pointerId:0,width:0,height:0,pressure:0,tangentialPressure:0,tiltX:0,tiltY:0,twist:0,pointerType:0,isPrimary:0}),fx=On(iT),sT=yt({},Sr,{submitter:0}),rT=On(sT),aT=yt({},Jl,{touches:0,targetTouches:0,changedTouches:0,altKey:0,metaKey:0,ctrlKey:0,shiftKey:0,getModifierState:Fm}),oT=On(aT),lT=yt({},Sr,{propertyName:0,elapsedTime:0,pseudoElement:0}),cT=On(lT),uT=yt({},ff,{deltaX:function(t){return"deltaX"in t?t.deltaX:"wheelDeltaX"in t?-t.wheelDeltaX:0},deltaY:function(t){return"deltaY"in t?t.deltaY:"wheelDeltaY"in t?-t.wheelDeltaY:"wheelDelta"in t?-t.wheelDelta:0},deltaZ:0,deltaMode:0}),fT=On(uT),hT=yt({},Sr,{newState:0,oldState:0,source:0}),dT=On(hT),pT=[9,13,27,32],zm=bs&&"CompositionEvent"in window,vl=null;bs&&"documentMode"in document&&(vl=document.documentMode);var mT=bs&&"TextEvent"in window&&!vl,o_=bs&&(!zm||vl&&8<vl&&11>=vl),hx=" ",dx=!1;function l_(t,e){switch(t){case"keyup":return pT.indexOf(e.keyCode)!==-1;case"keydown":return e.keyCode!==229;case"keypress":case"mousedown":case"focusout":return!0;default:return!1}}function c_(t){return t=t.detail,typeof t=="object"&&"data"in t?t.data:null}var Va=!1;function gT(t,e){switch(t){case"compositionend":return c_(e);case"keypress":return e.which!==32?null:(dx=!0,hx);case"textInput":return t=e.data,t===hx&&dx?null:t;default:return null}}function vT(t,e){if(Va)return t==="compositionend"||!zm&&l_(t,e)?(t=a_(),mu=Om=tr=null,Va=!1,t):null;switch(t){case"paste":return null;case"keypress":if(!(e.ctrlKey||e.altKey||e.metaKey)||e.ctrlKey&&e.altKey){if(e.char&&1<e.char.length)return e.char;if(e.which)return String.fromCharCode(e.which)}return null;case"compositionend":return o_&&e.locale!=="ko"?null:e.data;default:return null}}var xT={color:!0,date:!0,datetime:!0,"datetime-local":!0,email:!0,month:!0,number:!0,password:!0,range:!0,search:!0,tel:!0,text:!0,time:!0,url:!0,week:!0};function px(t){var e=t&&t.nodeName&&t.nodeName.toLowerCase();return e==="input"?!!xT[t.type]:e==="textarea"}function u_(t,e,n,i){Ga?ja?ja.push(i):ja=[i]:Ga=i,e=rf(e,"onChange"),0<e.length&&(n=new uf("onChange","change",null,n,i),t.push({event:n,listeners:e}))}var xl=null,Il=null;function yT(t){rA(t,0)}function hf(t){var e=pl(t);if(e_(e))return t}function mx(t,e){if(t==="change")return e}var f_=!1;bs&&(bs?($c="oninput"in document,$c||(jd=document.createElement("div"),jd.setAttribute("oninput","return;"),$c=typeof jd.oninput=="function"),jc=$c):jc=!1,f_=jc&&(!document.documentMode||9<document.documentMode));var jc,$c,jd;function gx(){xl&&(xl.detachEvent("onpropertychange",h_),Il=xl=null)}function h_(t){if(t.propertyName==="value"&&hf(Il)){var e=[];u_(e,Il,t,Lm(t)),r_(yT,e)}}function _T(t,e,n){t==="focusin"?(gx(),xl=e,Il=n,xl.attachEvent("onpropertychange",h_)):t==="focusout"&&gx()}function ST(t){if(t==="selectionchange"||t==="keyup"||t==="keydown")return hf(Il)}function AT(t,e){if(t==="click")return hf(e)}function MT(t,e){if(t==="input"||t==="change")return hf(e)}function ET(t,e){return t===e&&(t!==0||1/t===1/e)||t!==t&&e!==e}var ai=typeof Object.is=="function"?Object.is:ET;function Bl(t,e){if(ai(t,e))return!0;if(typeof t!="object"||t===null||typeof e!="object"||e===null)return!1;var n=Object.keys(t),i=Object.keys(e);if(n.length!==i.length)return!1;for(i=0;i<n.length;i++){var s=n[i];if(!Rp.call(e,s)||!ai(t[s],e[s]))return!1}return!0}function Np(t){if(t=t||(typeof document<"u"?document:void 0),typeof t>"u")return null;try{return t.activeElement||t.body}catch{return t.body}}function vx(t){for(;t&&t.firstChild;)t=t.firstChild;return t}function xx(t,e){var n=vx(t);t=0;for(var i;n;){if(n.nodeType===3){if(i=t+n.textContent.length,t<=e&&i>=e)return{node:n,offset:e-t};t=i}e:{for(;n;){if(n.nextSibling){n=n.nextSibling;break e}n=n.parentNode}n=void 0}n=vx(n)}}function d_(t,e){return t&&e?t===e?!0:t&&t.nodeType===3?!1:e&&e.nodeType===3?d_(t,e.parentNode):"contains"in t?t.contains(e):t.compareDocumentPosition?!!(t.compareDocumentPosition(e)&16):!1:!1}function p_(t){t=t!=null&&t.ownerDocument!=null&&t.ownerDocument.defaultView!=null?t.ownerDocument.defaultView:window;for(var e=Np(t.document);e instanceof t.HTMLIFrameElement;){try{var n=typeof e.contentWindow.location.href=="string"}catch{n=!1}if(n)t=e.contentWindow;else break;e=Np(t.document)}return e}function Hm(t){var e=t&&t.nodeName&&t.nodeName.toLowerCase();return e&&(e==="input"&&(t.type==="text"||t.type==="search"||t.type==="tel"||t.type==="url"||t.type==="password")||e==="textarea"||t.contentEditable==="true")}var TT=bs&&"documentMode"in document&&11>=document.documentMode,ka=null,Pp=null,yl=null,Lp=!1;function yx(t,e,n){var i=n.window===n?n.document:n.nodeType===9?n:n.ownerDocument;Lp||ka==null||ka!==Np(i)||(i=ka,"selectionStart"in i&&Hm(i)?i={start:i.selectionStart,end:i.selectionEnd}:(i=(i.ownerDocument&&i.ownerDocument.defaultView||window).getSelection(),i={anchorNode:i.anchorNode,anchorOffset:i.anchorOffset,focusNode:i.focusNode,focusOffset:i.focusOffset}),yl&&Bl(yl,i)||(yl=i,i=rf(Pp,"onSelect"),0<i.length&&(e=new uf("onSelect","select",null,e,n),t.push({event:e,listeners:i}),e.target=ka)))}function Vr(t,e){var n={};return n[t.toLowerCase()]=e.toLowerCase(),n["Webkit"+t]="webkit"+e,n["Moz"+t]="moz"+e,n}var Wa={animationend:Vr("Animation","AnimationEnd"),animationiteration:Vr("Animation","AnimationIteration"),animationstart:Vr("Animation","AnimationStart"),transitionrun:Vr("Transition","TransitionRun"),transitionstart:Vr("Transition","TransitionStart"),transitioncancel:Vr("Transition","TransitionCancel"),transitionend:Vr("Transition","TransitionEnd")},$d={},m_={};bs&&(m_=document.createElement("div").style,"AnimationEvent"in window||(delete Wa.animationend.animation,delete Wa.animationiteration.animation,delete Wa.animationstart.animation),"TransitionEvent"in window||delete Wa.transitionend.transition);function ua(t){if($d[t])return $d[t];if(!Wa[t])return t;var e=Wa[t],n;for(n in e)if(e.hasOwnProperty(n)&&n in m_)return $d[t]=e[n];return t}var g_=ua("animationend"),v_=ua("animationiteration"),x_=ua("animationstart"),bT=ua("transitionrun"),wT=ua("transitionstart"),CT=ua("transitioncancel"),y_=ua("transitionend"),__=new Map,Op="abort auxClick beforeToggle cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error fullscreenChange fullscreenError gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(" ");Op.push("scrollEnd");function Li(t,e){__.set(t,e),ca(e,[t])}var RT=0;function ws(t,e){if(t.name!=null&&t.name!=="auto")return t.name;if(e.autoName!==null)return e.autoName;t=Pi.identifierPrefix;var n=RT++;return t="_"+t+"t_"+n.toString(32)+"_",e.autoName=t}function _x(t){if(t==null||typeof t=="string")return t;var e=null,n=ro;if(n!==null)for(var i=0;i<n.length;i++){var s=t[n[i]];if(s!=null){if(s==="none")return"none";e=e==null?s:e+(" "+s)}}return e??t.default}function Bs(t,e){return t=_x(t),e=_x(e),e==null?t==="auto"?null:t:e==="auto"?null:e}var Ou=typeof reportError=="function"?reportError:function(t){if(typeof window=="object"&&typeof window.ErrorEvent=="function"){var e=new window.ErrorEvent("error",{bubbles:!0,cancelable:!0,message:typeof t=="object"&&t!==null&&typeof t.message=="string"?String(t.message):String(t),error:t});if(!window.dispatchEvent(e))return}else if(typeof process=="object"&&typeof process.emit=="function"){process.emit("uncaughtException",t);return}console.error(t)},gi=[],Xa=0,Gm=0;function df(){for(var t=Xa,e=Gm=Xa=0;e<t;){var n=gi[e];gi[e++]=null;var i=gi[e];gi[e++]=null;var s=gi[e];gi[e++]=null;var r=gi[e];if(gi[e++]=null,i!==null&&s!==null){var a=i.pending;a===null?s.next=s:(s.next=a.next,a.next=s),i.pending=s}r!==0&&S_(n,s,r)}}function pf(t,e,n,i){gi[Xa++]=t,gi[Xa++]=e,gi[Xa++]=n,gi[Xa++]=i,Gm|=i,t.lanes|=i,t=t.alternate,t!==null&&(t.lanes|=i)}function Vm(t,e,n,i){return pf(t,e,n,i),Fu(t)}function fa(t,e){return pf(t,null,null,e),Fu(t)}function S_(t,e,n){t.lanes|=n;var i=t.alternate;i!==null&&(i.lanes|=n);for(var s=!1,r=t.return;r!==null;)r.childLanes|=n,i=r.alternate,i!==null&&(i.childLanes|=n),r.tag===22&&(t=r.stateNode,t===null||t._visibility&1||(s=!0)),t=r,r=r.return;return t.tag===3?(r=t.stateNode,s&&e!==null&&(s=31-si(n),t=r.hiddenUpdates,i=t[s],i===null?t[s]=[e]:i.push(e),e.lane=n|536870912),r):null}function Fu(t){if(50<Rl)throw Rl=0,bu=null,Error(J(185));for(var e=t.return;e!==null;)t=e,e=t.return;return t.tag===3?t.stateNode:null}var Ya={};function DT(t,e,n,i){this.tag=t,this.key=n,this.sibling=this.child=this.return=this.stateNode=this.type=this.elementType=null,this.index=0,this.refCleanup=this.ref=null,this.pendingProps=e,this.dependencies=this.memoizedState=this.updateQueue=this.memoizedProps=null,this.mode=i,this.subtreeFlags=this.flags=0,this.deletions=null,this.childLanes=this.lanes=0,this.alternate=null}function Wn(t,e,n,i){return new DT(t,e,n,i)}function km(t){return t=t.prototype,!(!t||!t.isReactComponent)}function Es(t,e){var n=t.alternate;return n===null?(n=Wn(t.tag,e,t.key,t.mode),n.elementType=t.elementType,n.type=t.type,n.stateNode=t.stateNode,n.alternate=t,t.alternate=n):(n.pendingProps=e,n.type=t.type,n.flags=0,n.subtreeFlags=0,n.deletions=null),n.flags=t.flags&1206910976,n.childLanes=t.childLanes,n.lanes=t.lanes,n.child=t.child,n.memoizedProps=t.memoizedProps,n.memoizedState=t.memoizedState,n.updateQueue=t.updateQueue,e=t.dependencies,n.dependencies=e===null?null:{lanes:e.lanes,firstContext:e.firstContext},n.sibling=t.sibling,n.index=t.index,n.ref=t.ref,n.refCleanup=t.refCleanup,n}function A_(t,e){t.flags&=1206910978;var n=t.alternate;return n===null?(t.childLanes=0,t.lanes=e,t.child=null,t.subtreeFlags=0,t.memoizedProps=null,t.memoizedState=null,t.updateQueue=null,t.dependencies=null,t.stateNode=null):(t.childLanes=n.childLanes,t.lanes=n.lanes,t.child=n.child,t.subtreeFlags=0,t.deletions=null,t.memoizedProps=n.memoizedProps,t.memoizedState=n.memoizedState,t.updateQueue=n.updateQueue,t.type=n.type,e=n.dependencies,t.dependencies=e===null?null:{lanes:e.lanes,firstContext:e.firstContext}),t}function vu(t,e,n,i,s,r){var a=0;if(i=t,typeof i=="function")km(i)&&(a=1);else if(typeof i=="string")a=nw(t,n,rs.current)?26:t==="html"||t==="head"||t==="body"?27:5;else e:switch(i){case Ep:return t=Wn(31,n,e,s),t.elementType=Ep,t.lanes=r,t;case za:return Kr(n.children,s,r,e);case zy:a=8,s|=24;break;case Sp:return t=Wn(12,n,e,s|2),t.elementType=Sp,t.lanes=r,t;case Ap:return t=Wn(13,n,e,s),t.elementType=Ap,t.lanes=r,t;case Mp:return t=Wn(19,n,e,s),t.elementType=Mp,t.lanes=r,t;case vE:case Tp:return t=s|32,t=Wn(30,n,e,t),t.elementType=Tp,t.lanes=r,t.stateNode={autoName:null,paired:null,clones:null,ref:null},t;default:if(typeof i=="object"&&i!==null)switch(i.$$typeof){case ts:a=10;break e;case Hy:a=9;break e;case Dm:a=11;break e;case Um:a=14;break e;case Js:a=16,i=null;break e}a=29,n=Error(J(130,t===null?"null":typeof t,"")),i=null}return e=Wn(a,n,e,s),e.elementType=t,e.type=i,e.lanes=r,e}function Kr(t,e,n,i){return t=Wn(7,t,i,e),t.lanes=n,t}function ep(t,e,n){return t=Wn(6,t,null,e),t.lanes=n,t}function M_(t){var e=Wn(18,null,null,0);return e.stateNode=t,e}function tp(t,e,n){return e=Wn(4,t.children!==null?t.children:[],t.key,e),e.lanes=n,e.stateNode={containerInfo:t.containerInfo,pendingChildren:null,implementation:t.implementation},e}var Sx=new WeakMap;function _i(t,e){if(typeof t=="object"&&t!==null){var n=Sx.get(t);return n!==void 0?n:(e={value:t,source:e,stack:$v(e)},Sx.set(t,e),e)}return{value:t,source:e,stack:$v(e)}}var qa=[],Qa=0,zu=null,Nl=0,vi=[],xi=0,gr=null,is=1,ss="";function As(t,e){qa[Qa++]=Nl,qa[Qa++]=zu,zu=t,Nl=e}function E_(t,e,n){vi[xi++]=is,vi[xi++]=ss,vi[xi++]=gr,gr=t;var i=is;t=ss;var s=32-si(i)-1;i&=~(1<<s),n+=1;var r=32-si(e)+s;if(30<r){var a=s-s%5;r=(i&(1<<a)-1).toString(32),i>>=a,s-=a,is=1<<32-si(e)+s|n<<s|i,ss=r+t}else is=1<<r|n<<s|i,ss=t}function mf(t){t.return!==null&&(As(t,1),E_(t,1,0))}function Wm(t){for(;t===zu;)zu=qa[--Qa],qa[Qa]=null,Nl=qa[--Qa],qa[Qa]=null;for(;t===gr;)gr=vi[--xi],vi[xi]=null,ss=vi[--xi],vi[xi]=null,is=vi[--xi],vi[xi]=null}function T_(t,e){vi[xi++]=is,vi[xi++]=ss,vi[xi++]=gr,is=e.id,ss=e.overflow,gr=t}var fn=null,Tt=null,Ge=!1,or=null,Si=!1,Fp=Error(J(519));function vr(t){var e=Error(J(418,1<arguments.length&&arguments[1]!==void 0&&arguments[1]?"text":"HTML",""));throw Pl(_i(e,t)),Fp}function Ax(t){var e=t.stateNode,n=t.type,i=t.memoizedProps;switch(e[mn]=t,e[qn]=i,n){case"dialog":We("cancel",e),We("close",e);break;case"iframe":case"object":case"embed":We("load",e);break;case"video":case"audio":for(n=0;n<zl.length;n++)We(zl[n],e);break;case"source":We("error",e);break;case"img":case"image":case"link":We("error",e),We("load",e);break;case"details":We("toggle",e);break;case"input":We("invalid",e),t_(e,i.value,i.defaultValue,i.checked,i.defaultChecked,i.type,i.name,!0);break;case"select":We("invalid",e);break;case"textarea":We("invalid",e),i_(e,i.value,i.defaultValue,i.children)}n=i.children,typeof n!="string"&&typeof n!="number"&&typeof n!="bigint"||e.textContent===""+n||i.suppressHydrationWarning===!0||oA(e.textContent,n)?(i.popover!=null&&(We("beforetoggle",e),We("toggle",e)),i.onScroll!=null&&We("scroll",e),i.onScrollEnd!=null&&We("scrollend",e),i.onClick!=null&&(e.onclick=ns),e=!0):e=!1,e||vr(t,!0)}function Hu(t){for(fn=t.return;fn;)switch(fn.tag){case 5:case 31:case 13:Si=!1;return;case 27:case 3:Si=!0;return;default:fn=fn.return}}function Ba(t){if(t!==fn)return!1;if(!Ge)return Hu(t),Ge=!0,!1;var e=t.tag,n;if((n=e!==3&&e!==27)&&((n=e===5)&&(n=t.type,n=!(n!=="form"&&n!=="button")||Am(t.type,t.memoizedProps)),n=!n),n&&Tt&&vr(t),Hu(t),e===13){if(t=t.memoizedState,t=t!==null?t.dehydrated:null,!t)throw Error(J(317));Tt=xy(t)}else if(e===31){if(t=t.memoizedState,t=t!==null?t.dehydrated:null,!t)throw Error(J(317));Tt=xy(t)}else e===27?(e=Tt,Ar(t.type)?(t=bm,bm=null,Tt=t):Tt=e):Tt=fn?Ai(t.stateNode.nextSibling):null;return!0}function ea(){Tt=fn=null,Ge=!1}function np(){var t=or;return t!==null&&(Vn===null?Vn=t:Vn.push.apply(Vn,t),or=null),t}function Pl(t){or===null?or=[t]:or.push(t)}var zp=ls(null),ha=null,Ms=null;function nr(t,e,n){bt(zp,e._currentValue),e._currentValue=n}function Ts(t){t._currentValue=zp.current,vn(zp)}function xu(t,e,n){for(;t!==null;){var i=t.alternate;if((t.childLanes&e)!==e?(t.childLanes|=e,i!==null&&(i.childLanes|=e)):i!==null&&(i.childLanes&e)!==e&&(i.childLanes|=e),t===n)break;t=t.return}}function Hp(t,e,n,i){var s=t.child;for(s!==null&&(s.return=t);s!==null;){var r=s.dependencies;if(r!==null){var a=s.child;r=r.firstContext;e:for(;r!==null;){var o=r;r=s;for(var l=0;l<e.length;l++)if(o.context===e[l]){r.lanes|=n,o=r.alternate,o!==null&&(o.lanes|=n),xu(r.return,n,t),i||(a=null);break e}r=o.next}}else if(s.tag===18){if(a=s.return,a===null)throw Error(J(341));a.lanes|=n,r=a.alternate,r!==null&&(r.lanes|=n),xu(a,n,t),a=null}else s.tag===13&&s.memoizedState!==null&&s.memoizedState.dehydrated===null?(s.lanes|=n,a=s.alternate,a!==null&&(a.lanes|=n),xu(s.return,n,t),a=s.child,a=a!==null?a.sibling:null):a=s.child;if(a!==null)a.return=s;else for(a=s;a!==null;){if(a===t){a=null;break}if(s=a.sibling,s!==null){s.return=a.return,a=s;break}a=a.return}s=a}}function ta(t,e,n,i){t=null;for(var s=e,r=!1;s!==null;){if(!r){if((s.flags&524288)!==0)r=!0;else if((s.flags&262144)!==0)break}if(s.tag===10){var a=s.alternate;if(a===null)throw Error(J(387));if(a=a.memoizedProps,a!==null){var o=s.type;ai(s.pendingProps.value,a.value)||(t!==null?t.push(o):t=[o])}}else if(s===Iu.current){if(a=s.alternate,a===null)throw Error(J(387));a.memoizedState.memoizedState!==s.memoizedState.memoizedState&&(t!==null?t.push(vo):t=[vo])}s=s.return}return t!==null&&Hp(e,t,n,i),e.flags|=262144,t!==null}function Gu(t){for(t=t.firstContext;t!==null;){if(!ai(t.context._currentValue,t.memoizedValue))return!0;t=t.next}return!1}function na(t){ha=t,Ms=null,t=t.dependencies,t!==null&&(t.firstContext=null)}function gn(t){return b_(ha,t)}function eu(t,e){return ha===null&&na(t),b_(t,e)}function b_(t,e){var n=e._currentValue;if(e={context:e,memoizedValue:n,next:null},Ms===null){if(t===null)throw Error(J(308));Ms=e,t.dependencies={lanes:0,firstContext:e},t.flags|=524288}else Ms=Ms.next=e;return n}var UT=typeof AbortController<"u"?AbortController:function(){var t=[],e=this.signal={aborted:!1,addEventListener:function(n,i){t.push(i)}};this.abort=function(){e.aborted=!0,t.forEach(function(n){return n()})}},IT=sn.unstable_scheduleCallback,BT=sn.unstable_NormalPriority,Zt={$$typeof:ts,Consumer:null,Provider:null,_currentValue:null,_currentValue2:null,_threadCount:0};function Xm(){return{controller:new UT,data:new Map,refCount:0}}function jl(t){t.refCount--,t.refCount===0&&IT(BT,function(){t.controller.abort()})}function Mx(t,e){if((t.pendingLanes&4194048)!==0){var n=t.transitionTypes;for(n===null&&(n=t.transitionTypes=[]),t=0;t<e.length;t++){var i=e[t];n.indexOf(i)===-1&&n.push(i)}}}var ml=null;function NT(t){var e=t.transitionTypes;return t.transitionTypes=null,e}var _l=null,Gp=0,ia=0,$a=null;function PT(t,e){if(_l===null){var n=_l=[];Gp=0,ia=yg(),$a={status:"pending",value:void 0,then:function(i){n.push(i)}}}return Gp++,e.then(Ex,Ex),e}function Ex(){if(--Gp===0&&(ml=null,_l!==null)){$a!==null&&($a.status="fulfilled");var t=_l;_l=null,ia=0,$a=null;for(var e=0;e<t.length;e++)(0,t[e])()}}function LT(t,e){var n=[],i={status:"pending",value:null,reason:null,then:function(s){n.push(s)}};return t.then(function(){i.status="fulfilled",i.value=e;for(var s=0;s<n.length;s++)(0,n[s])(e)},function(s){for(i.status="rejected",i.reason=s,s=0;s<n.length;s++)(0,n[s])(void 0)}),i}var Tx=Ue.S;Ue.S=function(t,e){if(WS=ni(),typeof e=="object"&&e!==null&&typeof e.then=="function"&&PT(t,e),ml!==null)for(var n=po;n!==null;)Mx(n,ml),n=n.next;if(n=t.types,n!==null){for(var i=po;i!==null;)Mx(i,n),i=i.next;if(ia!==0){i=ml,i===null&&(i=ml=[]);for(var s=0;s<n.length;s++){var r=n[s];i.indexOf(r)===-1&&i.push(r)}}}Tx!==null&&Tx(t,e)};var Jr=ls(null);function Ym(){var t=Jr.current;return t!==null?t:xt.pooledCache}function yu(t,e){e===null?bt(Jr,Jr.current):bt(Jr,e.pool)}function w_(){var t=Ym();return t===null?null:{parent:Zt._currentValue,pool:t}}var Ao=Error(J(460)),qm=Error(J(474)),gf=Error(J(542)),Vu={then:function(){}};function bx(t){return t=t.status,t==="fulfilled"||t==="rejected"}function C_(t,e,n){switch(n=t[n],n===void 0?t.push(e):n!==e&&(e.then(ns,ns),e=n),e.status){case"fulfilled":return e.value;case"rejected":throw t=e.reason,Cx(t),t===void 0&&!("reason"in e)?Error(J(600)):t;default:if(typeof e.status=="string")e.then(ns,ns);else{if(t=xt,t!==null&&100<t.shellSuspendCounter)throw Error(J(482));t=e,t.status="pending",t.then(function(i){if(e.status==="pending"){var s=e;s.status="fulfilled",s.value=i}},function(i){if(e.status==="pending"){var s=e;s.status="rejected",s.reason=i}})}switch(e.status){case"fulfilled":return e.value;case"rejected":throw t=e.reason,Cx(t),t}throw jr=e,Ao}}function Xr(t){try{var e=t._init;return e(t._payload)}catch(n){throw n!==null&&typeof n=="object"&&typeof n.then=="function"?(jr=n,Ao):n}}var jr=null;function wx(){if(jr===null)throw Error(J(459));var t=jr;return jr=null,t}function Cx(t){if(t===Ao||t===gf)throw Error(J(483))}var eo=null,Ll=0;function tu(t){var e=Ll;return Ll+=1,eo===null&&(eo=[]),C_(eo,t,e)}function Qs(t,e){e=e.props.ref,t.ref=e!==void 0?e:null}function nu(t,e){throw e.$$typeof===gE?Error(J(525)):(t=Object.prototype.toString.call(e),Error(J(31,t==="[object Object]"?"object with keys {"+Object.keys(e).join(", ")+"}":t)))}function R_(t){function e(f,g){if(t){var S=f.deletions;S===null?(f.deletions=[g],f.flags|=16):S.push(g)}}function n(f,g){if(!t)return null;for(;g!==null;)e(f,g),g=g.sibling;return null}function i(f){for(var g=new Map;f!==null;)f.key===null?g.set(f.index,f):g.set(f.key,f),f=f.sibling;return g}function s(f,g){return f=Es(f,g),f.index=0,f.sibling=null,f}function r(f,g,S){return f.index=S,t?(S=f.alternate,S!==null?(S=S.index,S<g?(f.flags|=2,g):S):(f.flags|=134217730,g)):(f.flags|=1048576,g)}function a(f){return t&&f.alternate===null&&(f.flags|=134217730),f}function o(f,g,S,_){return g===null||g.tag!==6?(g=ep(S,f.mode,_),g.return=f,g):(g=s(g,S),g.return=f,g)}function l(f,g,S,_){var E=S.type;return E===za?(f=h(f,g,S.props.children,_,S.key),Qs(f,S),f):g!==null&&(g.elementType===E||typeof E=="object"&&E!==null&&E.$$typeof===Js&&Xr(E)===g.type)?(g=s(g,S.props),Qs(g,S),g.return=f,g):(g=vu(S.type,S.key,S.props,null,f.mode,_),Qs(g,S),g.return=f,g)}function c(f,g,S,_){return g===null||g.tag!==4||g.stateNode.containerInfo!==S.containerInfo||g.stateNode.implementation!==S.implementation?(g=tp(S,f.mode,_),g.return=f,g):(g=s(g,S.children||[]),g.return=f,g)}function h(f,g,S,_,E){return g===null||g.tag!==7?(g=Kr(S,f.mode,_,E),g.return=f,g):(g=s(g,S),g.return=f,g)}function p(f,g,S){if(typeof g=="string"&&g!==""||typeof g=="number"||typeof g=="bigint")return g=ep(""+g,f.mode,S),g.return=f,g;if(typeof g=="object"&&g!==null){switch(g.$$typeof){case Yc:return S=vu(g.type,g.key,g.props,null,f.mode,S),Qs(S,g),S.return=f,S;case hl:return g=tp(g,f.mode,S),g.return=f,g;case Js:return g=Xr(g),p(f,g,S)}if(dl(g)||al(g))return g=Kr(g,f.mode,S,null),g.return=f,g;if(typeof g.then=="function")return p(f,tu(g),S);if(g.$$typeof===ts)return p(f,eu(f,g),S);nu(f,g)}return null}function u(f,g,S,_){var E=g!==null?g.key:null;if(typeof S=="string"&&S!==""||typeof S=="number"||typeof S=="bigint")return E!==null?null:o(f,g,""+S,_);if(typeof S=="object"&&S!==null){switch(S.$$typeof){case Yc:return S.key===E?l(f,g,S,_):null;case hl:return S.key===E?c(f,g,S,_):null;case Js:return S=Xr(S),u(f,g,S,_)}if(dl(S)||al(S))return E!==null?null:h(f,g,S,_,null);if(typeof S.then=="function")return u(f,g,tu(S),_);if(S.$$typeof===ts)return u(f,g,eu(f,S),_);nu(f,S)}return null}function d(f,g,S,_,E){if(typeof _=="string"&&_!==""||typeof _=="number"||typeof _=="bigint")return f=f.get(S)||null,o(g,f,""+_,E);if(typeof _=="object"&&_!==null){switch(_.$$typeof){case Yc:return f=f.get(_.key===null?S:_.key)||null,l(g,f,_,E);case hl:return f=f.get(_.key===null?S:_.key)||null,c(g,f,_,E);case Js:return _=Xr(_),d(f,g,S,_,E)}if(dl(_)||al(_))return f=f.get(S)||null,h(g,f,_,E,null);if(typeof _.then=="function")return d(f,g,S,tu(_),E);if(_.$$typeof===ts)return d(f,g,S,eu(g,_),E);nu(g,_)}return null}function v(f,g,S,_){for(var E=null,T=null,C=g,y=g=0,b=null;C!==null&&y<S.length;y++){C.index>y?(b=C,C=null):b=C.sibling;var R=u(f,C,S[y],_);if(R===null){C===null&&(C=b);break}t&&C&&R.alternate===null&&e(f,C),g=r(R,g,y),T===null?E=R:T.sibling=R,T=R,C=b}if(y===S.length)return n(f,C),Ge&&As(f,y),E;if(C===null){for(;y<S.length;y++)C=p(f,S[y],_),C!==null&&(g=r(C,g,y),T===null?E=C:T.sibling=C,T=C);return Ge&&As(f,y),E}for(C=i(C);y<S.length;y++)b=d(C,f,y,S[y],_),b!==null&&(t&&(R=b.alternate,R!==null&&C.delete(R.key===null?y:R.key)),g=r(b,g,y),T===null?E=b:T.sibling=b,T=b);return t&&C.forEach(function(N){return e(f,N)}),Ge&&As(f,y),E}function M(f,g,S,_){if(S==null)throw Error(J(151));for(var E=null,T=null,C=g,y=g=0,b=null,R=S.next();C!==null&&!R.done;y++,R=S.next()){C.index>y?(b=C,C=null):b=C.sibling;var N=u(f,C,R.value,_);if(N===null){C===null&&(C=b);break}t&&C&&N.alternate===null&&e(f,C),g=r(N,g,y),T===null?E=N:T.sibling=N,T=N,C=b}if(R.done)return n(f,C),Ge&&As(f,y),E;if(C===null){for(;!R.done;y++,R=S.next())R=p(f,R.value,_),R!==null&&(g=r(R,g,y),T===null?E=R:T.sibling=R,T=R);return Ge&&As(f,y),E}for(C=i(C);!R.done;y++,R=S.next())R=d(C,f,y,R.value,_),R!==null&&(t&&(b=R.alternate,b!==null&&C.delete(b.key===null?y:b.key)),g=r(R,g,y),T===null?E=R:T.sibling=R,T=R);return t&&C.forEach(function(F){return e(f,F)}),Ge&&As(f,y),E}function m(f,g,S,_){if(typeof S=="object"&&S!==null&&S.type===za&&S.key===null&&S.props.ref===void 0&&(S=S.props.children),typeof S=="object"&&S!==null){switch(S.$$typeof){case Yc:e:{for(var E=S.key;g!==null;){if(g.key===E){if(E=S.type,E===za){if(g.tag===7){n(f,g.sibling),_=s(g,S.props.children),Qs(_,S),_.return=f,f=_;break e}}else if(g.elementType===E||typeof E=="object"&&E!==null&&E.$$typeof===Js&&Xr(E)===g.type){n(f,g.sibling),_=s(g,S.props),Qs(_,S),_.return=f,f=_;break e}n(f,g);break}else e(f,g);g=g.sibling}S.type===za?(_=Kr(S.props.children,f.mode,_,S.key),Qs(_,S),_.return=f,f=_):(_=vu(S.type,S.key,S.props,null,f.mode,_),Qs(_,S),_.return=f,f=_)}return a(f);case hl:e:{for(E=S.key;g!==null;){if(g.key===E)if(g.tag===4&&g.stateNode.containerInfo===S.containerInfo&&g.stateNode.implementation===S.implementation){n(f,g.sibling),_=s(g,S.children||[]),_.return=f,f=_;break e}else{n(f,g);break}else e(f,g);g=g.sibling}_=tp(S,f.mode,_),_.return=f,f=_}return a(f);case Js:return S=Xr(S),m(f,g,S,_)}if(dl(S))return v(f,g,S,_);if(al(S)){if(E=al(S),typeof E!="function")throw Error(J(150));return S=E.call(S),M(f,g,S,_)}if(typeof S.then=="function")return m(f,g,tu(S),_);if(S.$$typeof===ts)return m(f,g,eu(f,S),_);nu(f,S)}return typeof S=="string"&&S!==""||typeof S=="number"||typeof S=="bigint"?(S=""+S,g!==null&&g.tag===6?(n(f,g.sibling),_=s(g,S),_.return=f,f=_):(n(f,g),_=ep(S,f.mode,_),_.return=f,f=_),a(f)):n(f,g)}return function(f,g,S,_){try{Ll=0;var E=m(f,g,S,_);return eo=null,E}catch(C){if(C===Ao||C===gf)throw C;var T=Wn(29,C,null,f.mode);return T.lanes=_,T.return=f,T}}}var sa=R_(!0),D_=R_(!1),js=!1;function Qm(t){t.updateQueue={baseState:t.memoizedState,firstBaseUpdate:null,lastBaseUpdate:null,shared:{pending:null,lanes:0,hiddenCallbacks:null},callbacks:null}}function Vp(t,e){t=t.updateQueue,e.updateQueue===t&&(e.updateQueue={baseState:t.baseState,firstBaseUpdate:t.firstBaseUpdate,lastBaseUpdate:t.lastBaseUpdate,shared:t.shared,callbacks:null})}function lr(t){return{lane:t,tag:0,payload:null,callback:null,next:null}}function cr(t,e,n){var i=t.updateQueue;if(i===null)return null;if(i=i.shared,(st&2)!==0){var s=i.pending;return s===null?e.next=e:(e.next=s.next,s.next=e),i.pending=e,e=Fu(t),S_(t,null,n),e}return pf(t,i,e,n),Fu(t)}function Sl(t,e,n){if(e=e.updateQueue,e!==null&&(e=e.shared,(n&4194048)!==0)){var i=e.lanes;i&=t.pendingLanes,n|=i,e.lanes=n,qy(t,n)}}function ip(t,e){var n=t.updateQueue,i=t.alternate;if(i!==null&&(i=i.updateQueue,n===i)){var s=null,r=null;if(n=n.firstBaseUpdate,n!==null){do{var a={lane:n.lane,tag:n.tag,payload:n.payload,callback:null,next:null};r===null?s=r=a:r=r.next=a,n=n.next}while(n!==null);r===null?s=r=e:r=r.next=e}else s=r=e;n={baseState:i.baseState,firstBaseUpdate:s,lastBaseUpdate:r,shared:i.shared,callbacks:i.callbacks},t.updateQueue=n;return}t=n.lastBaseUpdate,t===null?n.firstBaseUpdate=e:t.next=e,n.lastBaseUpdate=e}var kp=!1;function Al(){if(kp){var t=$a;if(t!==null)throw t}}function Ml(t,e,n,i){kp=!1;var s=t.updateQueue;js=!1;var r=s.firstBaseUpdate,a=s.lastBaseUpdate,o=s.shared.pending;if(o!==null){s.shared.pending=null;var l=o,c=l.next;l.next=null,a===null?r=c:a.next=c,a=l;var h=t.alternate;h!==null&&(h=h.updateQueue,o=h.lastBaseUpdate,o!==a&&(o===null?h.firstBaseUpdate=c:o.next=c,h.lastBaseUpdate=l))}if(r!==null){var p=s.baseState;a=0,h=c=l=null,o=r;do{var u=o.lane&-536870913,d=u!==o.lane;if(d?(Ze&u)===u:(i&u)===u){u!==0&&u===ia&&(kp=!0),h!==null&&(h=h.next={lane:0,tag:o.tag,payload:o.payload,callback:null,next:null});e:{var v=t,M=o;u=e;var m=n;switch(M.tag){case 1:if(v=M.payload,typeof v=="function"){p=v.call(m,p,u);break e}p=v;break e;case 3:v.flags=v.flags&-65537|128;case 0:if(v=M.payload,u=typeof v=="function"?v.call(m,p,u):v,u==null)break e;p=yt({},p,u);break e;case 2:js=!0}}u=o.callback,u!==null&&(t.flags|=64,d&&(t.flags|=8192),d=s.callbacks,d===null?s.callbacks=[u]:d.push(u))}else d={lane:u,tag:o.tag,payload:o.payload,callback:o.callback,next:null},h===null?(c=h=d,l=p):h=h.next=d,a|=u;if(o=o.next,o===null){if(o=s.shared.pending,o===null)break;d=o,o=d.next,d.next=null,s.lastBaseUpdate=d,s.shared.pending=null}}while(!0);h===null&&(l=p),s.baseState=l,s.firstBaseUpdate=c,s.lastBaseUpdate=h,r===null&&(s.shared.lanes=0),_r|=a,t.lanes=a,t.memoizedState=p}}function U_(t,e){if(typeof t!="function")throw Error(J(191,t));t.call(e)}function I_(t,e){var n=t.callbacks;if(n!==null)for(t.callbacks=null,t=0;t<n.length;t++)U_(n[t],e)}var xr=ls(null),ku=ls(0);function Rx(t,e){t=Us,bt(ku,t),bt(xr,e),Us=t|e.baseLanes}function Wp(){bt(ku,Us),bt(xr,xr.current)}function Zm(){Us=ku.current,vn(xr),vn(ku)}var _n=ls(null),Tn=null;function ur(t){var e=t.alternate;bt(xn,xn.current&1),bt(_n,t),Tn===null&&(e===null||xr.current!==null||e.memoizedState!==null)&&(Tn=t)}function Xp(t){bt(xn,xn.current),bt(_n,t),Tn===null&&(Tn=t)}function B_(t){t.tag===22?(bt(xn,xn.current),bt(_n,t),Tn===null&&(Tn=t)):fr()}function fr(){bt(xn,xn.current),bt(_n,_n.current)}function $n(t){vn(_n),Tn===t&&(Tn=null),vn(xn)}var xn=ls(0);function Ol(t,e){bt(_n,_n.current),bt(xn,e)}function Km(t){vn(xn),vn(_n),Tn===t&&(Tn=null)}function Wu(t){for(var e=t;e!==null;){if(e.tag===13){var n=e.memoizedState;if(n!==null&&(n=n.dehydrated,n===null||Tm(n)||Mg(n)))return e}else if(e.tag===19&&e.memoizedProps.revealOrder!=="independent"){if((e.flags&128)!==0)return e}else if(e.child!==null){e.child.return=e,e=e.child;continue}if(e===t)break;for(;e.sibling===null;){if(e.return===null||e.return===t)return null;e=e.return}e.sibling.return=e.return,e=e.sibling}return null}var Cs=0,Oe=null,gt=null,Qt=null,Xu=!1,to=!1,ra=!1,Yu=0,Fl=0,no=null,OT=0;function zt(){throw Error(J(321))}function Jm(t,e){if(e===null)return!1;for(var n=0;n<e.length&&n<t.length;n++)if(!ai(t[n],e[n]))return!1;return!0}function jm(t,e,n,i,s,r){return Cs=r,Oe=e,e.memoizedState=null,e.updateQueue=null,e.lanes=0,Ue.H=t===null||t.memoizedState===null?uS:fS,ra=!1,r=n(i,s),ra=!1,to&&(r=P_(e,n,i,s)),N_(t),r}function N_(t){Ue.H=qu;var e=gt!==null&&gt.next!==null;if(Cs=0,Qt=gt=Oe=null,Xu=!1,Fl=0,no=null,e)throw Error(J(300));t===null||Kt||(t=t.dependencies,t!==null&&Gu(t)&&(Kt=!0))}function P_(t,e,n,i){Oe=t;var s=0;do{if(to&&(no=null),Fl=0,to=!1,25<=s)throw Error(J(301));if(s+=1,Qt=gt=null,t.updateQueue!=null){var r=t.updateQueue;r.lastEffect=null,r.events=null,r.stores=null,r.memoCache!=null&&(r.memoCache.index=0)}Ue.H=XT,r=e(n,i)}while(to);return r}function FT(){var t=Ue.H,e=t.useState()[0];return e=typeof e.then=="function"?$l(e):e,t=t.useState()[0],(gt!==null?gt.memoizedState:null)!==t&&(Oe.flags|=1024),e}function $m(){var t=Yu!==0;return Yu=0,t}function eg(t,e,n){e.updateQueue=t.updateQueue,e.flags&=-2053,t.lanes&=~n}function tg(t){if(Xu){for(t=t.memoizedState;t!==null;){var e=t.queue;e!==null&&(e.pending=null),t=t.next}Xu=!1}Cs=0,Qt=gt=Oe=null,to=!1,Fl=Yu=0,no=null}function Ln(){var t={memoizedState:null,baseState:null,baseQueue:null,queue:null,next:null};return Qt===null?Oe.memoizedState=Qt=t:Qt=Qt.next=t,Qt}function Vt(){if(gt===null){var t=Oe.alternate;t=t!==null?t.memoizedState:null}else t=gt.next;var e=Qt===null?Oe.memoizedState:Qt.next;if(e!==null)Qt=e,gt=t;else{if(t===null)throw Oe.alternate===null?Error(J(467)):Error(J(310));gt=t,t={memoizedState:gt.memoizedState,baseState:gt.baseState,baseQueue:gt.baseQueue,queue:gt.queue,next:null},Qt===null?Oe.memoizedState=Qt=t:Qt=Qt.next=t}return Qt}function vf(){return{lastEffect:null,events:null,stores:null,memoCache:null}}function $l(t){var e=Fl;return Fl+=1,no===null&&(no=[]),t=C_(no,t,e),e=Oe,(Qt===null?e.memoizedState:Qt.next)===null&&(e=e.alternate,Ue.H=e===null||e.memoizedState===null?uS:fS),t}function xf(t){if(t!==null&&typeof t=="object"){if(typeof t.then=="function")return $l(t);if(t.$$typeof===yE)return;if(t.$$typeof===ts)return gn(t)}throw Error(J(438,String(t)))}function ng(t){var e=null,n=Oe.updateQueue;if(n!==null&&(e=n.memoCache),e==null){var i=Oe.alternate;i!==null&&(i=i.updateQueue,i!==null&&(i=i.memoCache,i!=null&&(e={data:i.data.map(function(s){return s.slice()}),index:0})))}if(e==null&&(e={data:[],index:0}),n===null&&(n=vf(),Oe.updateQueue=n),n.memoCache=e,n=e.data[e.index],n===void 0)for(n=e.data[e.index]=Array(t),i=0;i<t;i++)n[i]=xE;return e.index++,n}function Rs(t,e){return typeof e=="function"?e(t):e}function _u(t){var e=Vt();return ig(e,gt,t)}function ig(t,e,n){var i=t.queue;if(i===null)throw Error(J(311));i.lastRenderedReducer=n;var s=t.baseQueue,r=i.pending;if(r!==null){if(s!==null){var a=s.next;s.next=r.next,r.next=a}e.baseQueue=s=r,i.pending=null}if(r=t.baseState,s===null)t.memoizedState=r;else{e=s.next;var o=a=null,l=null,c=e,h=!1;do{var p=c.lane&-536870913;if(p!==c.lane?(Ze&p)===p:(Cs&p)===p){var u=c.revertLane;if(u===0)l!==null&&(l=l.next={lane:0,revertLane:0,gesture:null,action:c.action,hasEagerState:c.hasEagerState,eagerState:c.eagerState,next:null}),p===ia&&(h=!0);else if((Cs&u)===u){c=c.next,u===ia&&(h=!0);continue}else p={lane:0,revertLane:c.revertLane,gesture:null,action:c.action,hasEagerState:c.hasEagerState,eagerState:c.eagerState,next:null},l===null?(o=l=p,a=r):l=l.next=p,Oe.lanes|=u,_r|=u;p=c.action,ra&&n(r,p),r=c.hasEagerState?c.eagerState:n(r,p)}else u={lane:p,revertLane:c.revertLane,gesture:c.gesture,action:c.action,hasEagerState:c.hasEagerState,eagerState:c.eagerState,next:null},l===null?(o=l=u,a=r):l=l.next=u,Oe.lanes|=p,_r|=p;c=c.next}while(c!==null&&c!==e);if(l===null?a=r:l.next=o,!ai(r,t.memoizedState)&&(Kt=!0,h&&(n=$a,n!==null)))throw n;t.memoizedState=r,t.baseState=a,t.baseQueue=l,i.lastRenderedState=r}return s===null&&(i.lanes=0),[t.memoizedState,i.dispatch]}function sp(t){var e=Vt(),n=e.queue;if(n===null)throw Error(J(311));n.lastRenderedReducer=t;var i=n.dispatch,s=n.pending,r=e.memoizedState;if(s!==null){n.pending=null;var a=s=s.next;do r=t(r,a.action),a=a.next;while(a!==s);ai(r,e.memoizedState)||(Kt=!0),e.memoizedState=r,e.baseQueue===null&&(e.baseState=r),n.lastRenderedState=r}return[r,i]}function L_(t,e,n){var i=Oe,s=Vt(),r=Ge;if(r){if(n===void 0)throw Error(J(407));n=n()}else n=e();var a=!ai((gt||s).memoizedState,n);if(a&&(s.memoizedState=n,Kt=!0),s=s.queue,sg(z_.bind(null,i,s,t),[t]),t=s.getSnapshot!==e||a||Qt!==null&&(Qt.memoizedState.tag&1)!==0,co(t?9:8,{destroy:void 0},F_.bind(null,i,s,n,e),null),t){if(i.flags|=2048,xt===null)throw Error(J(349));r||(Cs&127)!==0||O_(i,e,n)}return n}function O_(t,e,n){t.flags|=16384,t={getSnapshot:e,value:n},e=Oe.updateQueue,e===null?(e=vf(),Oe.updateQueue=e,e.stores=[t]):(n=e.stores,n===null?e.stores=[t]:n.push(t))}function F_(t,e,n,i){e.value=n,e.getSnapshot=i,H_(e)&&G_(t)}function z_(t,e,n){return n(function(){H_(e)&&G_(t)})}function H_(t){var e=t.getSnapshot;t=t.value;try{var n=e();return!ai(t,n)}catch{return!0}}function G_(t){var e=fa(t,2);e!==null&&Xn(e,t,2)}function Yp(t){var e=Ln();if(typeof t=="function"){var n=t;if(t=n(),ra){er(!0);try{n()}finally{er(!1)}}}return e.memoizedState=e.baseState=t,e.queue={pending:null,lanes:0,dispatch:null,lastRenderedReducer:Rs,lastRenderedState:t},e}function V_(t,e,n,i){return t.baseState=n,ig(t,gt,typeof i=="function"?i:Rs)}function zT(t,e,n,i,s){if(_f(t))throw Error(J(485));if(t=e.action,t!==null){var r={payload:s,action:t,next:null,isTransition:!0,status:"pending",value:null,reason:null,listeners:[],then:function(a){r.listeners.push(a)}};Ue.T!==null?n(!0):r.isTransition=!1,i(r),n=e.pending,n===null?(r.next=e.pending=r,k_(e,r)):(r.next=n.next,e.pending=n.next=r)}}function k_(t,e){var n=e.action,i=e.payload,s=t.state;if(e.isTransition){var r=Ue.T,a={};a.types=r!==null?r.types:null,Ue.T=a;try{var o=n(s,i),l=Ue.S;l!==null&&l(a,o),Dx(t,e,o)}catch(c){qp(t,e,c)}finally{r!==null&&a.types!==null&&(r.types=a.types),Ue.T=r}}else try{r=n(s,i),Dx(t,e,r)}catch(c){qp(t,e,c)}}function Dx(t,e,n){n!==null&&typeof n=="object"&&typeof n.then=="function"?n.then(function(i){Ux(t,e,i)},function(i){return qp(t,e,i)}):Ux(t,e,n)}function Ux(t,e,n){e.status="fulfilled",e.value=n,W_(e),t.state=n,e=t.pending,e!==null&&(n=e.next,n===e?t.pending=null:(n=n.next,e.next=n,k_(t,n)))}function qp(t,e,n){var i=t.pending;if(t.pending=null,i!==null){i=i.next;do e.status="rejected",e.reason=n,W_(e),e=e.next;while(e!==i)}t.action=null}function W_(t){t=t.listeners;for(var e=0;e<t.length;e++)(0,t[e])()}function X_(t,e){return e}function Ix(t,e){if(Ge){var n=xt.formState;if(n!==null){e:{var i=Oe;if(Ge){if(Tt){t:{for(var s=Tt,r=Si;s.nodeType!==8;){if(!r){s=null;break t}if(s=Ai(s.nextSibling),s===null){s=null;break t}}r=s.data,s=r==="F!"||r==="F"?s:null}if(s){Tt=Ai(s.nextSibling),i=s.data==="F!";break e}}vr(i)}i=!1}i&&(e=n[0])}}return n=Ln(),n.memoizedState=n.baseState=e,i={pending:null,lanes:0,dispatch:null,lastRenderedReducer:X_,lastRenderedState:e},n.queue=i,n=oS.bind(null,Oe,i),i.dispatch=n,i=Yp(!1),r=lg.bind(null,Oe,!1,i.queue),i=Ln(),s={state:e,dispatch:null,action:t,pending:null},i.queue=s,n=zT.bind(null,Oe,s,r,n),s.dispatch=n,i.memoizedState=t,[e,n,!1]}function Bx(t){var e=Vt();return Y_(e,gt,t)}function Y_(t,e,n){if(e=ig(t,e,X_)[0],t=_u(Rs)[0],typeof e=="object"&&e!==null&&typeof e.then=="function")try{var i=$l(e)}catch(a){throw a===Ao?gf:a}else i=e;e=Vt();var s=e.queue,r=s.dispatch;return n!==e.memoizedState&&(Oe.flags|=2048,co(9,{destroy:void 0},HT.bind(null,s,n),null)),[i,r,t]}function HT(t,e){t.action=e}function Nx(t){var e=Vt(),n=gt;if(n!==null)return Y_(e,n,t);Vt(),e=e.memoizedState,n=Vt();var i=n.queue.dispatch;return n.memoizedState=t,[e,i,!1]}function co(t,e,n,i){return t={tag:t,create:n,deps:i,inst:e,next:null},e=Oe.updateQueue,e===null&&(e=vf(),Oe.updateQueue=e),n=e.lastEffect,n===null?e.lastEffect=t.next=t:(i=n.next,n.next=t,t.next=i,e.lastEffect=t),t}function q_(){return Vt().memoizedState}function Su(t,e,n,i){var s=Ln();Oe.flags|=t,s.memoizedState=co(1|e,{destroy:void 0},n,i===void 0?null:i)}function yf(t,e,n,i){var s=Vt();i=i===void 0?null:i;var r=s.memoizedState.inst;gt!==null&&i!==null&&Jm(i,gt.memoizedState.deps)?s.memoizedState=co(e,r,n,i):(Oe.flags|=t,s.memoizedState=co(1|e,r,n,i))}function Px(t,e){Su(8390656,8,t,e)}function sg(t,e){yf(2048,8,t,e)}function GT(t){Oe.flags|=4;var e=Oe.updateQueue;if(e===null)e=vf(),Oe.updateQueue=e,e.events=[t];else{var n=e.events;n===null?e.events=[t]:n.push(t)}}function Q_(t){var e=Vt().memoizedState;return GT({ref:e,nextImpl:t}),function(){if((st&2)!==0)throw Error(J(440));return e.impl.apply(void 0,arguments)}}function Z_(t,e){return yf(4,2,t,e)}function K_(t,e){return yf(4,4,t,e)}function J_(t,e){if(typeof e=="function"){t=t();var n=e(t);return function(){typeof n=="function"?n():e(null)}}if(e!=null)return t=t(),e.current=t,function(){e.current=null}}function j_(t,e,n){n=n!=null?n.concat([t]):null,yf(4,4,J_.bind(null,e,t),n)}function rg(){}function $_(t,e){var n=Vt();e=e===void 0?null:e;var i=n.memoizedState;return e!==null&&Jm(e,i[1])?i[0]:(n.memoizedState=[t,e],t)}function eS(t,e){var n=Vt();e=e===void 0?null:e;var i=n.memoizedState;if(e!==null&&Jm(e,i[1]))return i[0];if(i=t(),ra){er(!0);try{t()}finally{er(!1)}}return n.memoizedState=[i,e],i}function ag(t,e,n){return n===void 0||(Cs&1073741824)!==0&&(Ze&261930)===0?t.memoizedState=e:(t.memoizedState=n,t=YS(),Oe.lanes|=t,_r|=t,n)}function tS(t,e,n,i){return ai(n,e)?n:xr.current!==null?(t=ag(t,n,i),ai(t,e)||(Kt=!0),t):(Cs&106)===0||(Cs&1073741824)!==0&&(Ze&261930)===0?(Kt=!0,t.memoizedState=n):(t=YS(),Oe.lanes|=t,_r|=t,e)}function nS(t,e,n,i,s){var r=rt.p;rt.p=r!==0&&8>r?r:8;var a=Ue.T,o={};o.types=a!==null?a.types:null,Ue.T=o,lg(t,!1,e,n);try{var l=s(),c=Ue.S;if(c!==null&&c(o,l),l!==null&&typeof l=="object"&&typeof l.then=="function"){var h=LT(l,i);El(t,e,h,ri(t))}else El(t,e,i,ri(t))}catch(p){El(t,e,{then:function(){},status:"rejected",reason:p},ri())}finally{rt.p=r,a!==null&&o.types!==null&&(a.types=o.types),Ue.T=a}}function VT(){}function Qp(t,e,n,i){if(t.tag!==5)throw Error(J(476));var s=iS(t).queue;nS(t,s,e,Zr,n===null?VT:function(){return sS(t),n(i)})}function iS(t){var e=t.memoizedState;if(e!==null)return e;e={memoizedState:Zr,baseState:Zr,baseQueue:null,queue:{pending:null,lanes:0,dispatch:null,lastRenderedReducer:Rs,lastRenderedState:Zr},next:null};var n={};return e.next={memoizedState:n,baseState:n,baseQueue:null,queue:{pending:null,lanes:0,dispatch:null,lastRenderedReducer:Rs,lastRenderedState:n},next:null},t.memoizedState=e,t=t.alternate,t!==null&&(t.memoizedState=e),e}function sS(t){var e=iS(t);e.next===null&&(e=t.alternate.memoizedState),El(t,e.next.queue,{},ri())}function og(){return gn(vo)}function rS(){return Vt().memoizedState}function aS(){return Vt().memoizedState}function kT(t){for(var e=t.return;e!==null;){switch(e.tag){case 24:case 3:var n=ri();t=lr(n);var i=cr(e,t,n);i!==null&&(Xn(i,e,n),Sl(i,e,n)),e={cache:Xm()},t.payload=e;return}e=e.return}}function WT(t,e,n){var i=ri();n={lane:i,revertLane:0,gesture:null,action:n,hasEagerState:!1,eagerState:null,next:null},_f(t)?lS(e,n):(n=Vm(t,e,n,i),n!==null&&(Xn(n,t,i),cS(n,e,i)))}function oS(t,e,n){var i=ri();El(t,e,n,i)}function El(t,e,n,i){var s={lane:i,revertLane:0,gesture:null,action:n,hasEagerState:!1,eagerState:null,next:null};if(_f(t))lS(e,s);else{var r=t.alternate;if(t.lanes===0&&(r===null||r.lanes===0)&&(r=e.lastRenderedReducer,r!==null))try{var a=e.lastRenderedState,o=r(a,n);if(s.hasEagerState=!0,s.eagerState=o,ai(o,a))return pf(t,e,s,0),xt===null&&df(),!1}catch{}if(n=Vm(t,e,s,i),n!==null)return Xn(n,t,i),cS(n,e,i),!0}return!1}function lg(t,e,n,i){if(i={lane:2,revertLane:yg(),gesture:null,action:i,hasEagerState:!1,eagerState:null,next:null},_f(t)){if(e)throw Error(J(479))}else e=Vm(t,n,i,2),e!==null&&Xn(e,t,2)}function _f(t){var e=t.alternate;return t===Oe||e!==null&&e===Oe}function lS(t,e){to=Xu=!0;var n=t.pending;n===null?e.next=e:(e.next=n.next,n.next=e),t.pending=e}function cS(t,e,n){if((n&4194048)!==0){var i=e.lanes;i&=t.pendingLanes,n|=i,e.lanes=n,qy(t,n)}}var qu={readContext:gn,use:xf,useCallback:zt,useContext:zt,useEffect:zt,useImperativeHandle:zt,useLayoutEffect:zt,useInsertionEffect:zt,useMemo:zt,useReducer:zt,useRef:zt,useState:zt,useDebugValue:zt,useDeferredValue:zt,useTransition:zt,useSyncExternalStore:zt,useId:zt,useHostTransitionStatus:zt,useFormState:zt,useActionState:zt,useOptimistic:zt,useMemoCache:zt,useCacheRefresh:zt,useEffectEvent:zt},uS={readContext:gn,use:xf,useCallback:function(t,e){return Ln().memoizedState=[t,e===void 0?null:e],t},useContext:gn,useEffect:Px,useImperativeHandle:function(t,e,n){n=n!=null?n.concat([t]):null,Su(4194308,4,J_.bind(null,e,t),n)},useLayoutEffect:function(t,e){return Su(4194308,4,t,e)},useInsertionEffect:function(t,e){Su(4,2,t,e)},useMemo:function(t,e){var n=Ln();e=e===void 0?null:e;var i=t();if(ra){er(!0);try{t()}finally{er(!1)}}return n.memoizedState=[i,e],i},useReducer:function(t,e,n){var i=Ln();if(n!==void 0){var s=n(e);if(ra){er(!0);try{n(e)}finally{er(!1)}}}else s=e;return i.memoizedState=i.baseState=s,t={pending:null,lanes:0,dispatch:null,lastRenderedReducer:t,lastRenderedState:s},i.queue=t,t=t.dispatch=WT.bind(null,Oe,t),[i.memoizedState,t]},useRef:function(t){var e=Ln();return t={current:t},e.memoizedState=t},useState:function(t){t=Yp(t);var e=t.queue,n=oS.bind(null,Oe,e);return e.dispatch=n,[t.memoizedState,n]},useDebugValue:rg,useDeferredValue:function(t,e){var n=Ln();return ag(n,t,e)},useTransition:function(){var t=Yp(!1);return t=nS.bind(null,Oe,t.queue,!0,!1),Ln().memoizedState=t,[!1,t]},useSyncExternalStore:function(t,e,n){var i=Oe,s=Ln();if(Ge){if(n===void 0)throw Error(J(407));n=n()}else{if(n=e(),xt===null)throw Error(J(349));(Ze&127)!==0||O_(i,e,n)}s.memoizedState=n;var r={value:n,getSnapshot:e};return s.queue=r,Px(z_.bind(null,i,r,t),[t]),i.flags|=2048,co(9,{destroy:void 0},F_.bind(null,i,r,n,e),null),n},useId:function(){var t=Ln(),e=xt.identifierPrefix;if(Ge){var n=ss,i=is;n=(i&~(1<<32-si(i)-1)).toString(32)+n,e="_"+e+"R_"+n,n=Yu++,0<n&&(e+="H"+n.toString(32)),e+="_"}else n=OT++,e="_"+e+"r_"+n.toString(32)+"_";return t.memoizedState=e},useHostTransitionStatus:og,useFormState:Ix,useActionState:Ix,useOptimistic:function(t){var e=Ln();e.memoizedState=e.baseState=t;var n={pending:null,lanes:0,dispatch:null,lastRenderedReducer:null,lastRenderedState:null};return e.queue=n,e=lg.bind(null,Oe,!0,n),n.dispatch=e,[t,e]},useMemoCache:ng,useCacheRefresh:function(){return Ln().memoizedState=kT.bind(null,Oe)},useEffectEvent:function(t){var e=Ln(),n={impl:t};return e.memoizedState=n,function(){if((st&2)!==0)throw Error(J(440));return n.impl.apply(void 0,arguments)}}},fS={readContext:gn,use:xf,useCallback:$_,useContext:gn,useEffect:sg,useImperativeHandle:j_,useInsertionEffect:Z_,useLayoutEffect:K_,useMemo:eS,useReducer:_u,useRef:q_,useState:function(){return _u(Rs)},useDebugValue:rg,useDeferredValue:function(t,e){var n=Vt();return tS(n,gt.memoizedState,t,e)},useTransition:function(){var t=_u(Rs)[0],e=Vt().memoizedState;return[typeof t=="boolean"?t:$l(t),e]},useSyncExternalStore:L_,useId:rS,useHostTransitionStatus:og,useFormState:Bx,useActionState:Bx,useOptimistic:function(t,e){var n=Vt();return V_(n,gt,t,e)},useMemoCache:ng,useCacheRefresh:aS,useEffectEvent:Q_},XT={readContext:gn,use:xf,useCallback:$_,useContext:gn,useEffect:sg,useImperativeHandle:j_,useInsertionEffect:Z_,useLayoutEffect:K_,useMemo:eS,useReducer:sp,useRef:q_,useState:function(){return sp(Rs)},useDebugValue:rg,useDeferredValue:function(t,e){var n=Vt();return gt===null?ag(n,t,e):tS(n,gt.memoizedState,t,e)},useTransition:function(){var t=sp(Rs)[0],e=Vt().memoizedState;return[typeof t=="boolean"?t:$l(t),e]},useSyncExternalStore:L_,useId:rS,useHostTransitionStatus:og,useFormState:Nx,useActionState:Nx,useOptimistic:function(t,e){var n=Vt();return gt!==null?V_(n,gt,t,e):(n.baseState=t,[t,n.queue.dispatch])},useMemoCache:ng,useCacheRefresh:aS,useEffectEvent:Q_};function rp(t,e,n,i){e=t.memoizedState,n=n(i,e),n=n==null?e:yt({},e,n),t.memoizedState=n,t.lanes===0&&(t.updateQueue.baseState=n)}var Zp={enqueueSetState:function(t,e,n){t=t._reactInternals;var i=ri(),s=lr(i);s.payload=e,n!=null&&(s.callback=n),e=cr(t,s,i),e!==null&&(Xn(e,t,i),Sl(e,t,i))},enqueueReplaceState:function(t,e,n){t=t._reactInternals;var i=ri(),s=lr(i);s.tag=1,s.payload=e,n!=null&&(s.callback=n),e=cr(t,s,i),e!==null&&(Xn(e,t,i),Sl(e,t,i))},enqueueForceUpdate:function(t,e){t=t._reactInternals;var n=ri(),i=lr(n);i.tag=2,e!=null&&(i.callback=e),e=cr(t,i,n),e!==null&&(Xn(e,t,n),Sl(e,t,n))}};function Lx(t,e,n,i,s,r,a){return t=t.stateNode,typeof t.shouldComponentUpdate=="function"?t.shouldComponentUpdate(i,r,a):e.prototype&&e.prototype.isPureReactComponent?!Bl(n,i)||!Bl(s,r):!0}function Ox(t,e,n,i){t=e.state,typeof e.componentWillReceiveProps=="function"&&e.componentWillReceiveProps(n,i),typeof e.UNSAFE_componentWillReceiveProps=="function"&&e.UNSAFE_componentWillReceiveProps(n,i),e.state!==t&&Zp.enqueueReplaceState(e,e.state,null)}function aa(t,e){var n=e;if("ref"in e){n={};for(var i in e)i!=="ref"&&(n[i]=e[i])}if(t=t.defaultProps){n===e&&(n=yt({},n));for(var s in t)n[s]===void 0&&(n[s]=t[s])}return n}function hS(t){Ou(t)}function dS(t){console.error(t)}function pS(t){Ou(t)}function Qu(t,e){try{var n=t.onUncaughtError;n(e.value,{componentStack:e.stack})}catch(i){setTimeout(function(){throw i})}}function Fx(t,e,n){try{var i=t.onCaughtError;i(n.value,{componentStack:n.stack,errorBoundary:e.tag===1?e.stateNode:null})}catch(s){setTimeout(function(){throw s})}}function Kp(t,e,n){return n=lr(n),n.tag=3,n.payload={element:null},n.callback=function(){Qu(t,e)},n}function mS(t){return t=lr(t),t.tag=3,t}function gS(t,e,n,i){var s=n.type.getDerivedStateFromError;if(typeof s=="function"){var r=i.value;t.payload=function(){return s(r)},t.callback=function(){Fx(e,n,i)}}var a=n.stateNode;a!==null&&typeof a.componentDidCatch=="function"&&(t.callback=function(){Fx(e,n,i),typeof s!="function"&&(hr===null?hr=new Set([this]):hr.add(this));var o=i.stack;this.componentDidCatch(i.value,{componentStack:o!==null?o:""})})}function YT(t,e,n,i,s){if(n.flags|=32768,i!==null&&typeof i=="object"&&typeof i.then=="function"){if(e=n.alternate,e!==null&&ta(e,n,s,!0),n=_n.current,n!==null){switch(n.tag){case 31:case 13:case 19:return Tn===null?nf():n.alternate===null&&Ht===0&&(Ht=3),n.flags&=-257,n.flags|=65536,n.lanes=s,i===Vu?n.flags|=16384:(e=n.updateQueue,e===null?n.updateQueue=new Set([i]):e.add(i),hp(t,i,s)),!1;case 22:return n.flags|=65536,i===Vu?n.flags|=16384:(e=n.updateQueue,e===null?(e={transitions:null,markerInstances:null,retryQueue:new Set([i])},n.updateQueue=e):(n=e.retryQueue,n===null?e.retryQueue=new Set([i]):n.add(i)),hp(t,i,s)),!1}throw Error(J(435,n.tag))}return hp(t,i,s),nf(),!1}if(Ge)return e=_n.current,e!==null?((e.flags&65536)===0&&(e.flags|=256),e.flags|=65536,e.lanes=s,i!==Fp&&(t=Error(J(422),{cause:i}),Pl(_i(t,n)))):(i!==Fp&&(e=Error(J(423),{cause:i}),Pl(_i(e,n))),t=t.current.alternate,t.flags|=65536,s&=-s,t.lanes|=s,i=_i(i,n),s=Kp(t.stateNode,i,s),ip(t,s),Ht!==4&&(Ht=2)),!1;var r=Error(J(520),{cause:i});if(r=_i(r,n),Cl===null?Cl=[r]:Cl.push(r),Ht!==4&&(Ht=2),e===null)return!0;i=_i(i,n),n=e;do{switch(n.tag){case 3:return n.flags|=65536,t=s&-s,n.lanes|=t,t=Kp(n.stateNode,i,t),ip(n,t),!1;case 1:if(e=n.type,r=n.stateNode,(n.flags&128)===0&&(typeof e.getDerivedStateFromError=="function"||r!==null&&typeof r.componentDidCatch=="function"&&(hr===null||!hr.has(r))))return n.flags|=65536,s&=-s,n.lanes|=s,s=mS(s),gS(s,t,n,i),ip(n,s),!1;break;case 22:if(n.memoizedState!==null)return n.flags|=65536,!1}n=n.return}while(n!==null);return!1}var cg=Error(J(461)),Kt=!1;function tn(t,e,n,i){e.child=t===null?D_(e,null,n,i):sa(e,t.child,n,i)}function zx(t,e,n,i,s){n=n.render;var r=e.ref;if("ref"in i){var a={};for(var o in i)o!=="ref"&&(a[o]=i[o])}else a=i;return na(e),i=jm(t,e,n,a,r,s),o=$m(),t!==null&&!Kt?(eg(t,e,s),Ds(t,e,s)):(Ge&&o&&mf(e),e.flags|=1,tn(t,e,i,s),e.child)}function Hx(t,e,n,i,s){if(t===null){var r=n.type;return typeof r=="function"&&!km(r)&&r.defaultProps===void 0&&n.compare===null?(e.tag=15,e.type=r,vS(t,e,r,i,s)):(t=vu(n.type,null,i,e,e.mode,s),t.ref=e.ref,t.return=e,e.child=t)}if(r=t.child,!fg(t,s)){var a=r.memoizedProps;if(n=n.compare,n=n!==null?n:Bl,n(a,i)&&t.ref===e.ref)return Ds(t,e,s)}return e.flags|=1,t=Es(r,i),t.ref=e.ref,t.return=e,e.child=t}function vS(t,e,n,i,s){if(t!==null){var r=t.memoizedProps;if(Bl(r,i)&&t.ref===e.ref)if(Kt=!1,e.pendingProps=i=r,fg(t,s))(t.flags&131072)!==0&&(Kt=!0);else return e.lanes=t.lanes,Ds(t,e,s)}return Jp(t,e,n,i,s)}function xS(t,e,n,i){var s=i.children,r=t!==null?t.memoizedState:null;if(t===null&&e.stateNode===null&&(e.stateNode={_visibility:1,_pendingMarkers:null,_retryCache:null,_transitions:null}),i.mode==="hidden"){if((e.flags&128)!==0){if(r=r!==null?r.baseLanes|n:n,t!==null){for(i=e.child=t.child,s=0;i!==null;)s=s|i.lanes|i.childLanes,i=i.sibling;i=s&~r}else i=0,e.child=null;return Gx(t,e,r,n,i)}if((n&536870912)!==0)e.memoizedState={baseLanes:0,cachePool:null},t!==null&&yu(e,r!==null?r.cachePool:null),r!==null?Rx(e,r):Wp(),B_(e);else return i=e.lanes=536870912,Gx(t,e,r!==null?r.baseLanes|n:n,n,i)}else r!==null?(yu(e,r.cachePool),Rx(e,r),fr(),e.memoizedState=null):(t!==null&&yu(e,null),Wp(),fr());return tn(t,e,s,n),e.child}function Tl(t,e){return t!==null&&t.tag===22||e.stateNode!==null||(e.stateNode={_visibility:1,_pendingMarkers:null,_retryCache:null,_transitions:null}),e.sibling}function Gx(t,e,n,i,s){var r=Ym();return r=r===null?null:{parent:Zt._currentValue,pool:r},e.memoizedState={baseLanes:n,cachePool:r},t!==null&&yu(e,null),Wp(),B_(e),t!==null&&ta(t,e,i,!0),e.childLanes=s,null}function Au(t,e){return e=Sf({mode:e.mode,children:e.children},t.mode),e.ref=t.ref,t.child=e,e.return=t,e}function Vx(t,e,n){return sa(e,t.child,null,n),t=Au(e,e.pendingProps),t.flags|=2,$n(e),e.memoizedState=null,t}function qT(t,e,n){var i=e.pendingProps,s=(e.flags&128)!==0;if(e.flags&=-129,t===null){if(Ge){if(i.mode==="hidden")return t=Au(e,i),e.lanes=536870912,t.memoizedState={baseLanes:0,cachePool:null},Tl(null,t);if(Xp(e),(t=Tt)?(t=xA(t,Si),t=t!==null&&t.data==="&"?t:null,t!==null&&(e.memoizedState={dehydrated:t,treeContext:gr!==null?{id:is,overflow:ss}:null,retryLane:536870912,hydrationErrors:null},n=M_(t),n.return=e,e.child=n,fn=e,Tt=null)):t=null,t===null)throw vr(e);return e.lanes=536870912,null}return Au(e,i)}var r=t.memoizedState;if(r!==null){var a=r.dehydrated;if(Xp(e),s)if(e.flags&256)e.flags&=-257,e=Vx(t,e,n);else if(e.memoizedState!==null)e.child=t.child,e.flags|=128,e=null;else throw Error(J(558));else if(Kt||ta(t,e,n,!1),s=(n&t.childLanes)!==0,Kt||s){if(xr.current===null){if(i=xt,i!==null&&(a=Qy(i,n),a!==0&&a!==r.retryLane))throw r.retryLane=a,fa(t,a),Xn(i,t,a),cg;nf()}e=Vx(t,e,n)}else t=r.treeContext,Tt=Ai(a.nextSibling),fn=e,Ge=!0,or=null,Si=!1,t!==null&&T_(e,t),e=Au(e,i),e.flags|=134221824;return e}return t=Es(t.child,{mode:i.mode,children:i.children}),t.ref=e.ref,e.child=t,t.return=e,t}function Pa(t,e){var n=e.ref;if(n===null)t!==null&&t.ref!==null&&(e.flags|=4194816);else{if(typeof n!="function"&&typeof n!="object")throw Error(J(284));(t===null||t.ref!==n)&&(e.flags|=4194816)}}function Jp(t,e,n,i,s){return na(e),n=jm(t,e,n,i,void 0,s),i=$m(),t!==null&&!Kt?(eg(t,e,s),Ds(t,e,s)):(Ge&&i&&mf(e),e.flags|=1,tn(t,e,n,s),e.child)}function kx(t,e,n,i,s,r){return na(e),e.updateQueue=null,n=P_(e,i,n,s),N_(t),i=$m(),t!==null&&!Kt?(eg(t,e,r),Ds(t,e,r)):(Ge&&i&&mf(e),e.flags|=1,tn(t,e,n,r),e.child)}function Wx(t,e,n,i,s){if(na(e),e.stateNode===null){var r=Ya,a=n.contextType;typeof a=="object"&&a!==null&&(r=gn(a)),r=new n(i,r),e.memoizedState=r.state!==null&&r.state!==void 0?r.state:null,r.updater=Zp,e.stateNode=r,r._reactInternals=e,r=e.stateNode,r.props=i,r.state=e.memoizedState,r.refs={},Qm(e),a=n.contextType,r.context=typeof a=="object"&&a!==null?gn(a):Ya,r.state=e.memoizedState,a=n.getDerivedStateFromProps,typeof a=="function"&&(rp(e,n,a,i),r.state=e.memoizedState),typeof n.getDerivedStateFromProps=="function"||typeof r.getSnapshotBeforeUpdate=="function"||typeof r.UNSAFE_componentWillMount!="function"&&typeof r.componentWillMount!="function"||(a=r.state,typeof r.componentWillMount=="function"&&r.componentWillMount(),typeof r.UNSAFE_componentWillMount=="function"&&r.UNSAFE_componentWillMount(),a!==r.state&&Zp.enqueueReplaceState(r,r.state,null),Ml(e,i,r,s),Al(),r.state=e.memoizedState),typeof r.componentDidMount=="function"&&(e.flags|=4194308),i=!0}else if(t===null){r=e.stateNode;var o=e.memoizedProps,l=aa(n,o);r.props=l;var c=r.context,h=n.contextType;a=Ya,typeof h=="object"&&h!==null&&(a=gn(h));var p=n.getDerivedStateFromProps;h=typeof p=="function"||typeof r.getSnapshotBeforeUpdate=="function",o=e.pendingProps!==o,h||typeof r.UNSAFE_componentWillReceiveProps!="function"&&typeof r.componentWillReceiveProps!="function"||(o||c!==a)&&Ox(e,r,i,a),js=!1;var u=e.memoizedState;r.state=u,Ml(e,i,r,s),Al(),c=e.memoizedState,o||u!==c||js?(typeof p=="function"&&(rp(e,n,p,i),c=e.memoizedState),(l=js||Lx(e,n,l,i,u,c,a))?(h||typeof r.UNSAFE_componentWillMount!="function"&&typeof r.componentWillMount!="function"||(typeof r.componentWillMount=="function"&&r.componentWillMount(),typeof r.UNSAFE_componentWillMount=="function"&&r.UNSAFE_componentWillMount()),typeof r.componentDidMount=="function"&&(e.flags|=4194308)):(typeof r.componentDidMount=="function"&&(e.flags|=4194308),e.memoizedProps=i,e.memoizedState=c),r.props=i,r.state=c,r.context=a,i=l):(typeof r.componentDidMount=="function"&&(e.flags|=4194308),i=!1)}else{r=e.stateNode,Vp(t,e),a=e.memoizedProps,h=aa(n,a),r.props=h,p=e.pendingProps,u=r.context,c=n.contextType,l=Ya,typeof c=="object"&&c!==null&&(l=gn(c)),o=n.getDerivedStateFromProps,(c=typeof o=="function"||typeof r.getSnapshotBeforeUpdate=="function")||typeof r.UNSAFE_componentWillReceiveProps!="function"&&typeof r.componentWillReceiveProps!="function"||(a!==p||u!==l)&&Ox(e,r,i,l),js=!1,u=e.memoizedState,r.state=u,Ml(e,i,r,s),Al();var d=e.memoizedState;a!==p||u!==d||js||t!==null&&t.dependencies!==null&&Gu(t.dependencies)?(typeof o=="function"&&(rp(e,n,o,i),d=e.memoizedState),(h=js||Lx(e,n,h,i,u,d,l)||t!==null&&t.dependencies!==null&&Gu(t.dependencies))?(c||typeof r.UNSAFE_componentWillUpdate!="function"&&typeof r.componentWillUpdate!="function"||(typeof r.componentWillUpdate=="function"&&r.componentWillUpdate(i,d,l),typeof r.UNSAFE_componentWillUpdate=="function"&&r.UNSAFE_componentWillUpdate(i,d,l)),typeof r.componentDidUpdate=="function"&&(e.flags|=4),typeof r.getSnapshotBeforeUpdate=="function"&&(e.flags|=1024)):(typeof r.componentDidUpdate!="function"||a===t.memoizedProps&&u===t.memoizedState||(e.flags|=4),typeof r.getSnapshotBeforeUpdate!="function"||a===t.memoizedProps&&u===t.memoizedState||(e.flags|=1024),e.memoizedProps=i,e.memoizedState=d),r.props=i,r.state=d,r.context=l,i=h):(typeof r.componentDidUpdate!="function"||a===t.memoizedProps&&u===t.memoizedState||(e.flags|=4),typeof r.getSnapshotBeforeUpdate!="function"||a===t.memoizedProps&&u===t.memoizedState||(e.flags|=1024),i=!1)}return r=i,Pa(t,e),i=(e.flags&128)!==0,r||i?(r=e.stateNode,n=i&&typeof n.getDerivedStateFromError!="function"?null:r.render(),e.flags|=1,t!==null&&i?(e.child=sa(e,t.child,null,s),e.child=sa(e,null,n,s)):tn(t,e,n,s),e.memoizedState=r.state,t=e.child):t=Ds(t,e,s),t}function Xx(t,e,n,i){return ea(),e.flags|=256,tn(t,e,n,i),e.child}var jp={dehydrated:null,treeContext:null,retryLane:0,hydrationErrors:null};function $p(t){return{baseLanes:t,cachePool:w_()}}function em(t,e,n){return t=t!==null?t.childLanes&~n:0,e&&(t|=ti),t}function yS(t,e,n){var i=e.pendingProps,s=!1,r=(e.flags&128)!==0,a;if((a=r)||(a=t!==null&&t.memoizedState===null?!1:(xn.current&2)!==0),a&&(s=!0,e.flags&=-129),a=(e.flags&32)!==0,e.flags&=-33,t===null){if(Ge){if(s?ur(e):fr(),(t=Tt)?(t=xA(t,Si),t=t!==null&&t.data!=="&"?t:null,t!==null&&(e.memoizedState={dehydrated:t,treeContext:gr!==null?{id:is,overflow:ss}:null,retryLane:536870912,hydrationErrors:null},n=M_(t),n.return=e,e.child=n,fn=e,Tt=null)):t=null,t===null)throw vr(e);return Mg(t)?e.lanes=32:e.lanes=536870912,null}return r=i.children,i=i.fallback,s?(fr(),s=e.mode,r=Sf({mode:"hidden",children:r},s),i=Kr(i,s,n,null),r.return=e,i.return=e,r.sibling=i,e.child=r,i=e.child,i.memoizedState=$p(n),i.childLanes=em(t,a,n),e.memoizedState=jp,Tl(null,i)):(ur(e),ug(e,r))}var o=t.memoizedState;if(o!==null){var l=o.dehydrated;if(l!==null)return QT(t,e,r,a,i,l,o,n)}return s?(fr(),s=i.fallback,r=e.mode,o=t.child,l=o.sibling,i=Es(o,{mode:"hidden",children:i.children}),i.subtreeFlags=o.subtreeFlags&1206910976,l!==null?s=Es(l,s):(s=Kr(s,r,n,null),s.flags|=2),s.return=e,i.return=e,i.sibling=s,e.child=i,Tl(null,i),i=e.child,s=t.child.memoizedState,s===null?s=$p(n):(r=s.cachePool,r!==null?(o=Zt._currentValue,r=r.parent!==o?{parent:o,pool:o}:r):r=w_(),s={baseLanes:s.baseLanes|n,cachePool:r}),i.memoizedState=s,i.childLanes=em(t,a,n),e.memoizedState=jp,Tl(t.child,i)):(ur(e),n=t.child,t=n.sibling,n=Es(n,{mode:"visible",children:i.children}),n.return=e,n.sibling=null,t!==null&&(a=e.deletions,a===null?(e.deletions=[t],e.flags|=16):a.push(t)),e.child=n,e.memoizedState=null,n)}function ug(t,e){return e=Sf({mode:"visible",children:e},t.mode),e.return=t,t.child=e}function Sf(t,e){return t=Wn(22,t,null,e),t.lanes=0,t}function iu(t,e,n){return sa(e,t.child,null,n),t=ug(e,e.pendingProps.children),t.flags|=2,e.memoizedState=null,t}function QT(t,e,n,i,s,r,a,o){if(n)return e.flags&256?(ur(e),e.flags&=-257,iu(t,e,o)):e.memoizedState!==null?(fr(),e.child=t.child,e.flags|=128,null):(fr(),r=s.fallback,a=e.mode,s=Sf({mode:"visible",children:s.children},a),r=Kr(r,a,o,null),r.flags|=2,s.return=e,r.return=e,s.sibling=r,e.child=s,sa(e,t.child,null,o),s=e.child,s.memoizedState=$p(o),s.childLanes=em(t,i,o),e.memoizedState=jp,Tl(null,s));if(ur(e),Mg(r)){if(i=r.nextSibling&&r.nextSibling.dataset,i)var l=i.dgst;return i=l,i!==""&&(s=Error(J(419)),s.stack="",s.digest=i,Pl({value:s,source:null,stack:null})),iu(t,e,o)}if(Kt||ta(t,e,o,!1),i=(o&t.childLanes)!==0,Kt||i){if(xr.current!==null)return iu(t,e,o);if(i=xt,i!==null&&(s=Qy(i,o),s!==0&&s!==a.retryLane))throw a.retryLane=s,fa(t,s),Xn(i,t,s),cg;return Tm(r)||nf(),iu(t,e,o)}return Tm(r)?(e.flags|=192,e.child=t.child,null):(t=a.treeContext,Tt=Ai(r.nextSibling),fn=e,Ge=!0,or=null,Si=!1,t!==null&&T_(e,t),e=ug(e,s.children),e.flags|=134221824,e)}function Yx(t,e,n){t.lanes|=e;var i=t.alternate;i!==null&&(i.lanes|=e),xu(t.return,e,n)}function qx(t){for(var e=null;t!==null;){var n=t.alternate;n!==null&&Wu(n)===null&&(e=t),t=t.sibling}return e}function su(t,e,n,i,s,r){var a=t.memoizedState;a===null?t.memoizedState={isBackwards:e,rendering:null,renderingStartTime:0,last:i,tail:n,tailMode:s,treeForkCount:r}:(a.isBackwards=e,a.rendering=null,a.renderingStartTime=0,a.last=i,a.tail=n,a.tailMode=s,a.treeForkCount=r)}function ap(t){var e=t.child;for(t.child=null;e!==null;){var n=e.sibling;e.sibling=t.child,t.child=e,e=n}}function tm(t,e,n){var i=e.pendingProps,s=i.revealOrder,r=i.tail;i=i.children;var a=xn.current;if(e.flags&128)return Ol(e,a),null;var o=(a&2)!==0;if(o?(a=a&1|2,e.flags|=128):a&=1,Ol(e,a),s==="backwards"&&t!==null?(ap(t),tn(t,e,i,n),ap(t)):tn(t,e,i,n),i=Ge?Nl:0,!o&&t!==null&&(t.flags&128)!==0)e:for(t=e.child;t!==null;){if(t.tag===13)t.memoizedState!==null&&Yx(t,n,e);else if(t.tag===19)Yx(t,n,e);else if(t.child!==null){t.child.return=t,t=t.child;continue}if(t===e)break e;for(;t.sibling===null;){if(t.return===null||t.return===e)break e;t=t.return}t.sibling.return=t.return,t=t.sibling}switch(s){case"backwards":n=qx(e.child),n===null?(s=e.child,e.child=null):(s=n.sibling,n.sibling=null,ap(e)),su(e,!0,s,null,r,i);break;case"unstable_legacy-backwards":for(n=null,s=e.child,e.child=null;s!==null;){if(t=s.alternate,t!==null&&Wu(t)===null){e.child=s;break}t=s.sibling,s.sibling=n,n=s,s=t}su(e,!0,n,null,r,i);break;case"together":su(e,!1,null,null,void 0,i);break;case"independent":e.memoizedState=null;break;default:n=qx(e.child),n===null?(s=e.child,e.child=null):(s=n.sibling,n.sibling=null),su(e,!1,s,n,r,i)}return e.child}function Qx(t,e,n){var i=e.pendingProps;return nr(e,e.type,i.value),tn(t,e,i.children,n),e.child}function Ds(t,e,n){if(t!==null&&(e.dependencies=t.dependencies),_r|=e.lanes,(n&e.childLanes)===0)if(t!==null){if(ta(t,e,n,!1),(n&e.childLanes)===0)return null}else return null;if(t!==null&&e.child!==t.child)throw Error(J(153));if(e.child!==null){for(t=e.child,n=Es(t,t.pendingProps),e.child=n,n.return=e;t.sibling!==null;)t=t.sibling,n=n.sibling=Es(t,t.pendingProps),n.return=e;n.sibling=null}return e.child}function fg(t,e){return(t.lanes&e)!==0?!0:(t=t.dependencies,!!(t!==null&&Gu(t)))}function ZT(t,e,n){switch(e.tag){case 3:Bu(e,e.stateNode.containerInfo),nr(e,Zt,t.memoizedState.cache),ea();break;case 27:case 5:Cp(e);break;case 4:Bu(e,e.stateNode.containerInfo);break;case 10:nr(e,e.type,e.memoizedProps.value);break;case 31:if(e.memoizedState!==null)return e.flags|=128,Xp(e),null;break;case 13:var i=e.memoizedState;if(i!==null){if(i.dehydrated!==null)return ur(e),e.flags|=128,null;i=ta(t,e,n,!1);var s=e.child.childLanes;return i||(n&s)!==0?yS(t,e,n):(ur(e),t=Ds(t,e,n),t!==null?t.sibling:null)}ur(e);break;case 19:if(e.flags&128)return tm(t,e,n);if(s=(t.flags&128)!==0,i=(n&e.childLanes)!==0,i||(ta(t,e,n,!1),i=(n&e.childLanes)!==0),s){if(i)return tm(t,e,n);e.flags|=128}if(s=e.memoizedState,s!==null&&(s.rendering=null,s.tail=null,s.lastEffect=null),Ol(e,xn.current),i)break;return null;case 22:return e.lanes=0,xS(t,e,n,e.pendingProps);case 24:nr(e,Zt,t.memoizedState.cache)}return Ds(t,e,n)}function _S(t,e,n){if(t!==null)if(t.memoizedProps!==e.pendingProps)Kt=!0;else{if(!fg(t,n)&&(e.flags&128)===0)return Kt=!1,ZT(t,e,n);Kt=(t.flags&131072)!==0}else Kt=!1,Ge&&(e.flags&1048576)!==0&&E_(e,Nl,e.index);switch(e.lanes=0,e.tag){case 16:e:{var i=e.pendingProps;if(t=Xr(e.elementType),e.type=t,typeof t=="function")km(t)?(i=aa(t,i),e.tag=1,e=Wx(null,e,t,i,n)):(e.tag=0,e=Jp(null,e,t,i,n));else{if(t!=null){var s=t.$$typeof;if(s===Dm){e.tag=11,e=zx(null,e,t,i,n);break e}else if(s===Um){e.tag=14,e=Hx(null,e,t,i,n);break e}else if(s===ts){e.tag=10,e.type=t,e=Qx(null,e,n);break e}}throw e=bp(t)||t,Error(J(306,e,""))}}return e;case 0:return Jp(t,e,e.type,e.pendingProps,n);case 1:return i=e.type,s=aa(i,e.pendingProps),Wx(t,e,i,s,n);case 3:e:{if(Bu(e,e.stateNode.containerInfo),t===null)throw Error(J(387));i=e.pendingProps;var r=e.memoizedState;s=r.element,Vp(t,e),Ml(e,i,null,n);var a=e.memoizedState;if(i=a.cache,nr(e,Zt,i),i!==r.cache&&Hp(e,[Zt],n,!0),Al(),i=a.element,r.isDehydrated)if(r={element:i,isDehydrated:!1,cache:a.cache},e.updateQueue.baseState=r,e.memoizedState=r,e.flags&256){e=Xx(t,e,i,n);break e}else if(i!==s){s=_i(Error(J(424)),e),Pl(s),e=Xx(t,e,i,n);break e}else for(t=e.stateNode.containerInfo,t.nodeType===9?t=t.body:t=t.nodeName==="HTML"?t.ownerDocument.body:t,Tt=Ai(t.firstChild),fn=e,Ge=!0,or=null,Si=!0,n=D_(e,null,i,n),e.child=n;n;)n.flags=n.flags&-3|134221824,n=n.sibling;else{if(ea(),i===s){e=Ds(t,e,n);break e}tn(t,e,i,n)}e=e.child}return e;case 26:return Pa(t,e),t===null?(n=Sy(e.type,null,e.pendingProps,null))?e.memoizedState=n:Ge||(e.stateNode=cA(e.type,e.pendingProps,ar.current,e)):e.memoizedState=Sy(e.type,t.memoizedProps,e.pendingProps,t.memoizedState),null;case 27:return Cp(e),t===null&&Ge&&(i=e.stateNode=yA(e.type,e.pendingProps,ar.current),fn=e,Si=!0,s=Tt,Ar(e.type)?(bm=s,Tt=Ai(i.firstChild)):Tt=s),tn(t,e,e.pendingProps.children,n),Pa(t,e),t===null&&(e.flags|=4194304),e.child;case 5:return t===null&&Ge&&((s=i=Tt)&&(i=Gb(i,e.type,e.pendingProps,Si),i!==null?(e.stateNode=i,fn=e,Tt=Ai(i.firstChild),Si=!1,s=!0):s=!1),s||vr(e)),Cp(e),s=e.type,r=e.pendingProps,a=t!==null?t.memoizedProps:null,i=r.children,Am(s,r)?i=null:a!==null&&Am(s,a)&&(e.flags|=32),e.memoizedState!==null&&(s=jm(t,e,FT,null,null,n),vo._currentValue=s),Pa(t,e),tn(t,e,i,n),e.child;case 6:return t===null&&Ge&&((t=n=Tt)&&(n=Vb(n,e.pendingProps,Si),n!==null?(e.stateNode=n,fn=e,Tt=null,t=!0):t=!1),t||vr(e)),null;case 13:return yS(t,e,n);case 4:return Bu(e,e.stateNode.containerInfo),i=e.pendingProps,t===null?e.child=sa(e,null,i,n):tn(t,e,i,n),e.child;case 11:return zx(t,e,e.type,e.pendingProps,n);case 7:return i=e.pendingProps,Pa(t,e),tn(t,e,i,n),e.child;case 8:return tn(t,e,e.pendingProps.children,n),e.child;case 12:return tn(t,e,e.pendingProps.children,n),e.child;case 10:return Qx(t,e,n);case 9:return s=e.type._context,i=e.pendingProps.children,na(e),s=gn(s),i=i(s),e.flags|=1,tn(t,e,i,n),e.child;case 14:return Hx(t,e,e.type,e.pendingProps,n);case 15:return vS(t,e,e.type,e.pendingProps,n);case 19:return tm(t,e,n);case 31:return qT(t,e,n);case 22:return xS(t,e,n,e.pendingProps);case 24:return na(e),i=gn(Zt),t===null?(s=Ym(),s===null&&(s=xt,r=Xm(),s.pooledCache=r,r.refCount++,r!==null&&(s.pooledCacheLanes|=n),s=r),e.memoizedState={parent:i,cache:s},Qm(e),nr(e,Zt,s)):((t.lanes&n)!==0&&(Vp(t,e),Ml(e,null,null,n),Al()),s=t.memoizedState,r=e.memoizedState,s.parent!==i?(s={parent:i,cache:i},e.memoizedState=s,e.lanes===0&&(e.memoizedState=e.updateQueue.baseState=s),nr(e,Zt,i)):(i=r.cache,nr(e,Zt,i),i!==s.cache&&Hp(e,[Zt],n,!0))),tn(t,e,e.pendingProps.children,n),e.child;case 30:return e.stateNode===null&&(e.stateNode={autoName:null,paired:null,clones:null,ref:null}),i=e.pendingProps,i.name!=null&&i.name!=="auto"?e.flags|=t===null?18882560:18874368:Ge&&mf(e),t!==null&&t.memoizedProps.name!==i.name?e.flags|=4194816:Pa(t,e),tn(t,e,i.children,n),e.child;case 29:throw e.pendingProps}throw Error(J(156,e.tag))}function Ss(t){t.flags|=4}function op(t,e,n,i,s){var r;if((r=(t.mode&32)!==0)&&(r=n===null?Ey(e,i):Ey(e,i)&&(i.src!==n.src||i.srcSet!==n.srcSet)),r){if(t.flags|=16777216,(s&335544128)===s)if(t.stateNode.complete)t.flags|=8192;else if(ZS())t.flags|=8192;else throw jr=Vu,qm}else t.flags&=-16777217}function Zx(t,e){if(e.type!=="stylesheet"||(e.state.loading&4)!==0)t.flags&=-16777217;else if(t.flags|=16777216,!MA(e))if(ZS())t.flags|=8192;else throw jr=Vu,qm}function ru(t,e){e!==null&&(t.flags|=4),t.flags&16384&&(e=t.tag!==22?Xy():536870912,t.lanes|=e,uo|=e)}function ll(t,e){if(!Ge)switch(t.tailMode){case"visible":break;case"collapsed":for(var n=t.tail,i=null;n!==null;)n.alternate!==null&&(i=n),n=n.sibling;i===null?e||t.tail===null?t.tail=null:t.tail.sibling=null:i.sibling=null;break;default:for(e=t.tail,n=null;e!==null;)e.alternate!==null&&(n=e),e=e.sibling;n===null?t.tail=null:n.sibling=null}}function Et(t){var e=t.alternate!==null&&t.alternate.child===t.child,n=0,i=0;if(e)for(var s=t.child;s!==null;)n|=s.lanes|s.childLanes,i|=s.subtreeFlags&1206910976,i|=s.flags&1206910976,s.return=t,s=s.sibling;else for(s=t.child;s!==null;)n|=s.lanes|s.childLanes,i|=s.subtreeFlags,i|=s.flags,s.return=t,s=s.sibling;return t.subtreeFlags|=i,t.childLanes=n,e}function KT(t,e,n){var i=e.pendingProps;switch(Wm(e),e.tag){case 16:case 15:case 0:case 11:case 7:case 8:case 12:case 9:case 14:return Et(e),null;case 1:return Et(e),null;case 3:return n=e.stateNode,i=null,t!==null&&(i=t.memoizedState.cache),e.memoizedState.cache!==i&&(e.flags|=2048),Ts(Zt),ao(),n.pendingContext&&(n.context=n.pendingContext,n.pendingContext=null),(t===null||t.child===null)&&(Ba(e)?Ss(e):t===null||t.memoizedState.isDehydrated&&(e.flags&256)===0||(e.flags|=1024,np())),Et(e),null;case 26:var s=e.type,r=e.memoizedState;return t===null?(Ss(e),r!==null?(Et(e),Zx(e,r)):(Et(e),op(e,s,null,i,n))):r?r!==t.memoizedState?(Ss(e),Et(e),Zx(e,r)):(Et(e),e.flags&=-16777217):(t=t.memoizedProps,t!==i&&Ss(e),Et(e),op(e,s,t,i,n)),null;case 27:if(Nu(e),n=ar.current,s=e.type,t!==null&&e.stateNode!=null)t.memoizedProps!==i&&Ss(e);else{if(!i){if(e.stateNode===null)throw Error(J(166));return Et(e),e.subtreeFlags&=-33554433,null}t=rs.current,Ba(e)?Ax(e,t):(t=yA(s,i,n),e.stateNode=t,Ss(e))}return Et(e),e.subtreeFlags&=-33554433,null;case 5:if(Nu(e),s=e.type,t!==null&&e.stateNode!=null)t.memoizedProps!==i&&Ss(e);else{if(!i){if(e.stateNode===null)throw Error(J(166));return Et(e),e.subtreeFlags&=-33554433,null}if(r=rs.current,Ba(e))Ax(e,r);else{var a=Gl(ar.current);switch(r){case 1:r=a.createElementNS("http://www.w3.org/2000/svg",s);break;case 2:r=a.createElementNS("http://www.w3.org/1998/Math/MathML",s);break;default:switch(s){case"svg":r=a.createElementNS("http://www.w3.org/2000/svg",s);break;case"math":r=a.createElementNS("http://www.w3.org/1998/Math/MathML",s);break;case"script":r=a.createElement("div"),r.innerHTML="<script><\/script>",r=r.removeChild(r.firstChild);break;case"select":r=typeof i.is=="string"?a.createElement("select",{is:i.is}):a.createElement("select"),i.multiple?r.multiple=!0:i.size&&(r.size=i.size);break;default:r=typeof i.is=="string"?a.createElement(s,{is:i.is}):a.createElement(s)}}r[mn]=e,r[qn]=i;e:for(a=e.child;a!==null;){if(a.tag===5||a.tag===6)r.appendChild(a.stateNode);else if(a.tag!==4&&a.tag!==27&&a.child!==null){a.child.return=a,a=a.child;continue}if(a===e)break e;for(;a.sibling===null;){if(a.return===null||a.return===e)break e;a=a.return}a.sibling.return=a.return,a=a.sibling}e.stateNode=r;e:switch(yn(r,s,i),s){case"button":case"input":case"select":case"textarea":i=!!i.autoFocus;break e;case"img":i=!0;break e;default:i=!1}i&&Ss(e)}}return Et(e),e.subtreeFlags&=-33554433,op(e,e.type,t===null?null:t.memoizedProps,e.pendingProps,n),null;case 6:if(t&&e.stateNode!=null)t.memoizedProps!==i&&Ss(e);else{if(typeof i!="string"&&e.stateNode===null)throw Error(J(166));if(t=ar.current,Ba(e)){if(t=e.stateNode,n=e.memoizedProps,i=null,s=fn,s!==null)switch(s.tag){case 27:case 5:i=s.memoizedProps}t[mn]=e,t=!!(t.nodeValue===n||i!==null&&i.suppressHydrationWarning===!0||oA(t.nodeValue,n)),t||vr(e,!0)}else t=Gl(t).createTextNode(i),t[mn]=e,e.stateNode=t}return Et(e),null;case 31:if(n=e.memoizedState,t===null||t.memoizedState!==null){if(i=Ba(e),n!==null){if(t===null){if(!i)throw Error(J(318));if(t=e.memoizedState,t=t!==null?t.dehydrated:null,!t)throw Error(J(557));t[mn]=e}else ea(),(e.flags&128)===0&&(e.memoizedState=null),e.flags|=4;Et(e),t=!1}else n=np(),t!==null&&t.memoizedState!==null&&(t.memoizedState.hydrationErrors=n),t=!0;if(!t)return e.flags&256?($n(e),e):($n(e),null);if((e.flags&128)!==0)throw Error(J(558))}return Et(e),null;case 13:if(i=e.memoizedState,t===null||t.memoizedState!==null&&t.memoizedState.dehydrated!==null){if(s=Ba(e),i!==null&&i.dehydrated!==null){if(t===null){if(!s)throw Error(J(318));if(s=e.memoizedState,s=s!==null?s.dehydrated:null,!s)throw Error(J(317));s[mn]=e}else ea(),(e.flags&128)===0&&(e.memoizedState=null),e.flags|=4;Et(e),s=!1}else s=np(),t!==null&&t.memoizedState!==null&&(t.memoizedState.hydrationErrors=s),s=!0;if(!s)return e.flags&256?($n(e),e):($n(e),null)}return $n(e),(e.flags&128)!==0?(e.lanes=n,e):(n=i!==null,t=t!==null&&t.memoizedState!==null,n&&(i=e.child,s=null,i.alternate!==null&&i.alternate.memoizedState!==null&&i.alternate.memoizedState.cachePool!==null&&(s=i.alternate.memoizedState.cachePool.pool),r=null,i.memoizedState!==null&&i.memoizedState.cachePool!==null&&(r=i.memoizedState.cachePool.pool),r!==s&&(i.flags|=2048)),n!==t&&n&&(e.child.flags|=8192),ru(e,e.updateQueue),Et(e),null);case 4:return ao(),t===null&&_g(e.stateNode.containerInfo),e.flags|=67108864,Et(e),null;case 10:return Ts(e.type),Et(e),null;case 19:if(Km(e),i=e.memoizedState,i===null)return Et(e),null;if(s=(e.flags&128)!==0,r=i.rendering,r===null)if(s)ll(i,!1);else{if(Ht!==0||t!==null&&(t.flags&128)!==0)for(t=e.child;t!==null;){if(r=Wu(t),r!==null){for(e.flags|=128,ll(i,!1),t=r.updateQueue,e.updateQueue=t,ru(e,t),e.subtreeFlags=0,t=n,n=e.child;n!==null;)A_(n,t),n=n.sibling;return Ol(e,xn.current&1|2),Ge&&As(e,i.treeForkCount),e.child}t=t.sibling}i.tail!==null&&ni()>ef&&(e.flags|=128,s=!0,ll(i,!1),e.lanes=4194304)}else{if(!s)if(t=Wu(r),t!==null){if(e.flags|=128,s=!0,t=t.updateQueue,e.updateQueue=t,ru(e,t),ll(i,!0),i.tail===null&&i.tailMode!=="collapsed"&&i.tailMode!=="visible"&&!r.alternate&&!Ge)return Et(e),null}else 2*ni()-i.renderingStartTime>ef&&n!==536870912&&(e.flags|=128,s=!0,ll(i,!1),e.lanes=4194304);i.isBackwards?(r.sibling=e.child,e.child=r):(t=i.last,t!==null?t.sibling=r:e.child=r,i.last=r)}if(i.tail!==null){t=i.tail;e:{for(n=t;n!==null;){if(n.alternate!==null){n=!1;break e}n=n.sibling}n=!0}return i.rendering=t,i.tail=t.sibling,i.renderingStartTime=ni(),t.sibling=null,r=xn.current,r=s?r&1|2:r&1,i.tailMode==="visible"||i.tailMode==="collapsed"||!n||Ge?Ol(e,r):(n=r,bt(_n,e),bt(xn,n),Tn===null&&(Tn=e)),Ge&&As(e,i.treeForkCount),t}return Et(e),null;case 22:case 23:return $n(e),Zm(),i=e.memoizedState!==null,t!==null?t.memoizedState!==null!==i&&(e.flags|=8192):i&&(e.flags|=8192),i?(n&536870912)!==0&&(e.flags&128)===0&&(Et(e),e.subtreeFlags&6&&(e.flags|=8192)):Et(e),n=e.updateQueue,n!==null&&ru(e,n.retryQueue),n=null,t!==null&&t.memoizedState!==null&&t.memoizedState.cachePool!==null&&(n=t.memoizedState.cachePool.pool),i=null,e.memoizedState!==null&&e.memoizedState.cachePool!==null&&(i=e.memoizedState.cachePool.pool),i!==n&&(e.flags|=2048),t!==null&&vn(Jr),null;case 24:return n=null,t!==null&&(n=t.memoizedState.cache),e.memoizedState.cache!==n&&(e.flags|=2048),Ts(Zt),Et(e),null;case 25:return null;case 30:return e.flags|=33554432,Et(e),null}throw Error(J(156,e.tag))}function JT(t,e){switch(Wm(e),e.tag){case 1:return t=e.flags,t&65536?(e.flags=t&-65537|128,e):null;case 3:return Ts(Zt),ao(),t=e.flags,(t&65536)!==0&&(t&128)===0?(e.flags=t&-65537|128,e):null;case 26:case 27:case 5:return Nu(e),null;case 31:if(e.memoizedState!==null){if($n(e),e.alternate===null)throw Error(J(340));ea()}return t=e.flags,t&65536?(e.flags=t&-65537|128,e):null;case 13:if($n(e),t=e.memoizedState,t!==null&&t.dehydrated!==null){if(e.alternate===null)throw Error(J(340));ea()}return t=e.flags,t&65536?(e.flags=t&-65537|128,e):null;case 19:return Km(e),t=e.flags,t&65536?(e.flags=t&-65537|128,t=e.memoizedState,t!==null&&(t.rendering=null,t.tail=null),e.flags|=4,e):null;case 4:return ao(),null;case 10:return Ts(e.type),null;case 22:case 23:return $n(e),Zm(),t!==null&&vn(Jr),t=e.flags,t&65536?(e.flags=t&-65537|128,e):null;case 24:return Ts(Zt),null;case 25:return null;default:return null}}function SS(t,e){switch(Wm(e),e.tag){case 3:Ts(Zt),ao();break;case 26:case 27:case 5:Nu(e);break;case 4:ao();break;case 31:e.memoizedState!==null&&$n(e);break;case 13:$n(e);break;case 19:Km(e);break;case 10:Ts(e.type);break;case 22:case 23:$n(e),Zm(),t!==null&&vn(Jr);break;case 24:Ts(Zt)}}function ec(t,e){try{var n=e.updateQueue,i=n!==null?n.lastEffect:null;if(i!==null){var s=i.next;n=s;do{if((n.tag&t)===t){i=void 0;var r=n.create,a=n.inst;i=r(),a.destroy=i}n=n.next}while(n!==s)}}catch(o){pt(e,e.return,o)}}function yr(t,e,n){try{var i=e.updateQueue,s=i!==null?i.lastEffect:null;if(s!==null){var r=s.next;i=r;do{if((i.tag&t)===t){var a=i.inst,o=a.destroy;if(o!==void 0){a.destroy=void 0,s=e;var l=n,c=o;try{c()}catch(h){pt(s,l,h)}}}i=i.next}while(i!==r)}}catch(h){pt(e,e.return,h)}}function AS(t){var e=t.updateQueue;if(e!==null){var n=t.stateNode;try{I_(e,n)}catch(i){pt(t,t.return,i)}}}function MS(t,e,n){n.props=aa(t.type,t.memoizedProps),n.state=t.memoizedState;try{n.componentWillUnmount()}catch(i){pt(t,e,i)}}function $i(t,e){try{var n=t.ref;if(n!==null){switch(t.tag){case 26:case 27:case 5:var i=t.stateNode;break;case 30:var s=t.stateNode,r=ws(t.memoizedProps,s);(s.ref===null||s.ref.name!==r)&&(s.ref=dA(r)),i=s.ref;break;case 7:if(t.stateNode===null){var a=new oi(t);Yn(t.child,!1,zb,a,void 0,void 0),t.stateNode=a}i=t.stateNode;break;default:i=t.stateNode}typeof n=="function"?t.refCleanup=n(i):n.current=i}}catch(o){pt(t,e,o)}}function pn(t,e){var n=t.ref,i=t.refCleanup;if(n!==null)if(typeof i=="function")try{i()}catch(s){pt(t,e,s)}finally{t.refCleanup=null,t=t.alternate,t!=null&&(t.refCleanup=null)}else if(typeof n=="function")try{n(null)}catch(s){pt(t,e,s)}else n.current=null}function Zu(t,e){if((t.tag===5||t.tag===27||t.tag===6)&&t.alternate===null&&e!==null)for(var n=0;n<e.length;n++)vA(t.stateNode,e[n])}function Kx(t){for(var e=t.return;e!==null&&(dg(e)&&vA(t.stateNode,e.stateNode),!hg(e));)e=e.return}function bl(t){for(var e=t.return;e!==null&&(dg(e)&&Hb(t.stateNode,e.stateNode),!hg(e));)e=e.return}function hg(t){return t.tag===5||t.tag===3||t.tag===27}function dg(t){return t&&t.tag===7&&t.stateNode!==null}function nm(t){var e=t.type,n=t.memoizedProps,i=t.stateNode;try{e:switch(e){case"button":case"input":case"select":case"textarea":n.autoFocus&&i.focus();break e;case"img":n.src?i.src=n.src:n.srcSet&&(i.srcset=n.srcSet)}}catch(s){pt(t,t.return,s)}}function lp(t,e,n){try{var i=t.stateNode;Sb(i,t.type,n,e),i[qn]=e}catch(s){pt(t,t.return,s)}}function ES(t){return t.tag===5||t.tag===3||t.tag===26||t.tag===27&&Ar(t.type)||t.tag===4}function cp(t){e:for(;;){for(;t.sibling===null;){if(t.return===null||ES(t.return))return null;t=t.return}for(t.sibling.return=t.return,t=t.sibling;t.tag!==5&&t.tag!==6&&t.tag!==18;){if(t.tag===27&&Ar(t.type)||t.flags&2||t.child===null||t.tag===4)continue e;t.child.return=t,t=t.child}if(!(t.flags&2))return t.stateNode}}function im(t,e,n,i){var s=t.tag;if(s===5||s===6)s=t.stateNode,e?(n.nodeType===9?n.body:n.nodeName==="HTML"?n.ownerDocument.body:n).insertBefore(s,e):(e=n.nodeType===9?n.body:n.nodeName==="HTML"?n.ownerDocument.body:n,e.appendChild(s),n=n._reactRootContainer,n!=null||e.onclick!==null||(e.onclick=ns)),Zu(t,i),nt=!0;else if(s!==4&&(s===27&&(Zu(t,i),i=null,Ar(t.type)&&(n=t.stateNode,e=null)),t=t.child,t!==null))for(im(t,e,n,i),t=t.sibling;t!==null;)im(t,e,n,i),t=t.sibling}function Ku(t,e,n,i){var s=t.tag;if(s===5||s===6)s=t.stateNode,e?n.insertBefore(s,e):n.appendChild(s),Zu(t,i),nt=!0;else if(s!==4&&(s===27&&(Zu(t,i),i=null,Ar(t.type)&&(n=t.stateNode)),t=t.child,t!==null))for(Ku(t,e,n,i),t=t.sibling;t!==null;)Ku(t,e,n,i),t=t.sibling}function TS(t){var e=t.stateNode,n=t.memoizedProps;try{for(var i=t.type,s=e.attributes;s.length;)e.removeAttributeNode(s[0]);yn(e,i,n),e[mn]=t,e[qn]=n}catch(r){pt(t,t.return,r)}}var Ju=!1,ei=null;function Jx(t){(t.tag===30||(t.subtreeFlags&33554432)!==0)&&(Ju=!0)}var es=null;function jx(){var t=es;return es=null,t}var kn=0;function Mo(t,e,n,i,s){return kn=0,bS(t.child,e,n,i,s)}function bS(t,e,n,i,s){for(var r=!1;t!==null;){if(t.tag===5){var a=t.stateNode;if(i!==null){var o=Mm(a);i.push(o),o.view&&(r=!0)}else r||Mm(a).view&&(r=!0);Ju=!0,uA(a,kn===0?e:e+"_"+kn,n),kn++}else(t.tag!==22||t.memoizedState===null)&&(t.tag===30&&s||bS(t.child,e,n,i,s)&&(r=!0));t=t.sibling}return r}function os(t,e){for(;t!==null;)t.tag===5?fA(t.stateNode,t.memoizedProps):(t.tag!==22||t.memoizedState===null)&&(t.tag===30&&e||os(t.child,e)),t=t.sibling}function Mu(t){if((t.subtreeFlags&18874368)!==0)for(t=t.child;t!==null;){if((t.tag!==22||t.memoizedState===null)&&(Mu(t),t.tag===30&&(t.flags&18874368)!==0&&t.stateNode.paired)){var e=t.memoizedProps;if(e.name==null||e.name==="auto")throw Error(J(544));var n=e.name;e=Bs(e.default,e.share),e!=="none"&&(Mo(t,n,e,null,!1)||os(t.child,!1))}t=t.sibling}}function sm(t,e){if(t.tag===30){var n=t.stateNode,i=t.memoizedProps,s=ws(i,n),r=Bs(i.default,n.paired?i.share:i.enter);r!=="none"?Mo(t,s,r,null,!1)?(Mu(t),n.paired||e||fo(t,i.onEnter)):os(t.child,!1):Mu(t)}else if((t.subtreeFlags&33554432)!==0)for(t=t.child;t!==null;)sm(t,e),t=t.sibling;else Mu(t)}function rm(t){if(ei!==null&&ei.size!==0){var e=ei;if((t.subtreeFlags&18874368)!==0)for(t=t.child;t!==null;){if(t.tag!==22||t.memoizedState===null){if(t.tag===30&&(t.flags&18874368)!==0){var n=t.memoizedProps,i=n.name;if(i!=null&&i!=="auto"){var s=e.get(i);if(s!==void 0){var r=Bs(n.default,n.share);if(r!=="none"&&(Mo(t,i,r,null,!1)?(r=t.stateNode,s.paired=r,r.paired=s,fo(t,n.onShare)):os(t.child,!1)),e.delete(i),e.size===0)break}}}rm(t)}t=t.sibling}}}function am(t){if(t.tag===30){var e=t.memoizedProps,n=ws(e,t.stateNode),i=ei!==null?ei.get(n):void 0,s=Bs(e.default,i!==void 0?e.share:e.exit);s!=="none"&&(Mo(t,n,s,null,!1)?i!==void 0?(s=t.stateNode,i.paired=s,s.paired=i,ei.delete(n),fo(t,e.onShare)):fo(t,e.onExit):os(t.child,!1)),ei!==null&&rm(t)}else if((t.subtreeFlags&33554432)!==0)for(t=t.child;t!==null;)am(t),t=t.sibling;else ei!==null&&rm(t)}function wS(t){for(t=t.child;t!==null;){if(t.tag===30){var e=t.memoizedProps,n=ws(e,t.stateNode);e=Bs(e.default,e.update),t.flags&=-5,e!=="none"&&Mo(t,n,e,t.memoizedState=[],!1)}else(t.subtreeFlags&33554432)!==0&&wS(t);t=t.sibling}}function om(t){if((t.subtreeFlags&18874368)!==0)for(t=t.child;t!==null;){if(t.tag!==22||t.memoizedState===null){if(t.tag===30&&(t.flags&18874368)!==0){var e=t.stateNode;e.paired!==null&&(e.paired=null,os(t.child,!1))}om(t)}t=t.sibling}}function Eu(t){if(t.tag===30)t.stateNode.paired=null,os(t.child,!1),om(t);else if((t.subtreeFlags&33554432)!==0)for(t=t.child;t!==null;)Eu(t),t=t.sibling;else om(t)}function CS(t){for(t=t.child;t!==null;)t.tag===30?os(t.child,!1):(t.subtreeFlags&33554432)!==0&&CS(t),t=t.sibling}function pg(t,e,n,i,s,r,a){for(var o=!1;e!==null;){if(e.tag===5){var l=e.stateNode;if(r!==null&&kn<r.length){var c=r[kn],h=Mm(l);(c.view||h.view)&&(o=!0);var p;if(p=(t.flags&4)===0)if(h.clip)p=!0;else{p=c.rect;var u=h.rect;p=p.y!==u.y||p.x!==u.x||p.height!==u.height||p.width!==u.width}p&&(t.flags|=4),h.abs?h=!c.abs:(c=c.rect,h=h.rect,h=c.height!==h.height||c.width!==h.width),h&&(t.flags|=32)}else t.flags|=32;(t.flags&4)!==0&&uA(l,kn===0?n:n+"_"+kn,s),o&&(t.flags&4)!==0||(es===null&&(es=[]),es.push(l,kn===0?i:i+"_"+kn,e.memoizedProps)),kn++}else(e.tag!==22||e.memoizedState===null)&&(e.tag===30&&a?t.flags|=e.flags&32:pg(t,e.child,n,i,s,r,a)&&(o=!0));e=e.sibling}return o}function RS(t,e){for(t=t.child;t!==null;){if(t.tag===30){var n=t.memoizedProps,i=t.stateNode,s=ws(n,i),r=Bs(n.default,n.update);if(e){i=i.clones;var a=i===null?null:i.map(wb)}else a=t.memoizedState,t.memoizedState=null;i=t;var o=t.child;kn=0,s=pg(i,o,s,s,r,a,!1),(t.flags&4)!==0&&s&&(e||fo(t,n.onUpdate))}else(t.subtreeFlags&33554432)!==0&&RS(t,e);t=t.sibling}}var ln=!1,lt=!1,Ki=!1,up=!1,$x=typeof WeakSet=="function"?WeakSet:Set,cn=null,Ji=!1,gl=!1,ju=!1,lm=!1;function jT(t,e,n){if(t=t.containerInfo,_m=xo,t=p_(t),Hm(t)){if("selectionStart"in t)var i={start:t.selectionStart,end:t.selectionEnd};else e:{i=(i=t.ownerDocument)&&i.defaultView||window;var s=i.getSelection&&i.getSelection();if(s&&s.rangeCount!==0){i=s.anchorNode;var r=s.anchorOffset,a=s.focusNode;s=s.focusOffset;try{i.nodeType,a.nodeType}catch{i=null;break e}var o=0,l=-1,c=-1,h=0,p=0,u=t,d=null;t:for(;;){for(var v;u!==i||r!==0&&u.nodeType!==3||(l=o+r),u!==a||s!==0&&u.nodeType!==3||(c=o+s),u.nodeType===3&&(o+=u.nodeValue.length),(v=u.firstChild)!==null;)d=u,u=v;for(;;){if(u===t)break t;if(d===i&&++h===r&&(l=o),d===a&&++p===s&&(c=o),(v=u.nextSibling)!==null)break;u=d,d=u.parentNode}u=v}i=l===-1||c===-1?null:{start:l,end:c}}else i=null}i=i||{start:0,end:0}}else i=null;for(Sm={focusedElem:t,selectionRange:i},xo=!1,n=(n&335544064)===n,cn=e,e=n?9270:1024;cn!==null;){if(t=cn,n&&(i=t.deletions,i!==null))for(r=0;r<i.length;r++)n&&am(i[r]);if(t.alternate===null&&(t.flags&2)!==0)n&&Jx(t),au(n);else{if(t.tag===22){if(i=t.alternate,t.memoizedState!==null){i!==null&&i.memoizedState===null&&n&&am(i),au(n);continue}else if(i!==null&&i.memoizedState!==null){n&&Jx(t),au(n);continue}}i=t.child,(t.subtreeFlags&e)!==0&&i!==null?(i.return=t,cn=i):(n&&wS(t),au(n))}}ei=null}function au(t){for(;cn!==null;){var e=cn,n=t,i=e.alternate,s=e.flags;switch(e.tag){case 0:case 11:case 15:break;case 1:if((s&1024)!==0&&i!==null){n=void 0,s=i.memoizedProps,i=i.memoizedState;var r=e.stateNode;try{var a=aa(e.type,s);n=r.getSnapshotBeforeUpdate(a,i),r.__reactInternalSnapshotBeforeUpdate=n}catch(o){pt(e,e.return,o)}}break;case 3:if((s&1024)!==0){if(i=e.stateNode.containerInfo,n=i.nodeType,n===9)Em(i);else if(n===1)switch(i.nodeName){case"HEAD":case"HTML":case"BODY":Em(i);break;default:i.textContent=""}}break;case 5:case 26:case 27:case 6:case 4:case 17:break;case 30:n&&i!==null&&(n=ws(i.memoizedProps,i.stateNode),s=e.memoizedProps,s=Bs(s.default,s.update),s!=="none"&&Mo(i,n,s,i.memoizedState=[],!0));break;default:if((s&1024)!==0)throw Error(J(163))}if(i=e.sibling,i!==null){i.return=e.return,cn=i;break}cn=e.return}}function DS(t,e,n){var i=n.flags;switch(n.tag){case 0:case 11:case 15:ji(t,n),i&4&&ec(5,n);break;case 1:if(ji(t,n),i&4)if(t=n.stateNode,e===null)try{t.componentDidMount()}catch(a){pt(n,n.return,a)}else{var s=aa(n.type,e.memoizedProps);e=e.memoizedState;try{t.componentDidUpdate(s,e,t.__reactInternalSnapshotBeforeUpdate)}catch(a){pt(n,n.return,a)}}i&64&&AS(n),i&512&&$i(n,n.return);break;case 3:if(ji(t,n),i&64&&(t=n.updateQueue,t!==null)){if(e=null,n.child!==null)switch(n.child.tag){case 27:case 5:e=n.child.stateNode;break;case 1:e=n.child.stateNode}try{I_(t,e)}catch(a){pt(n,n.return,a)}}break;case 27:e===null&&i&4&&TS(n);case 26:case 5:ji(t,n),e===null&&i&4&&nm(n),i&512&&$i(n,n.return);break;case 12:ji(t,n);break;case 31:ji(t,n),i&4&&NS(t,n);break;case 13:ji(t,n),i&4&&PS(t,n),i&64&&(t=n.memoizedState,t!==null&&(t=t.dehydrated,t!==null&&(n=ub.bind(null,n),kb(t,n))));break;case 22:if(i=n.memoizedState!==null||ln,!i){var r=e!==null&&e.memoizedState!==null||lt;e=ln,s=lt,ln=i,(lt=r)&&!s?(i=2,(n.subtreeFlags&8772)!==0&&(i|=1),Ii(t,n,i)):ji(t,n),ln=e,lt=s}break;case 30:ji(t,n),i&512&&$i(n,n.return);break;case 7:i&512&&$i(n,n.return);default:ji(t,n)}}function cm(t,e){for(t=t.child;t!==null;)US(t,e),t=t.sibling}function US(t,e){switch(t.tag){case 5:case 26:try{var n=t.stateNode;if(e){var i=n.style;typeof i.setProperty=="function"?i.setProperty("display","none","important"):i.display="none"}else{var s=t.stateNode,r=t.memoizedProps.style,a=r!=null&&r.hasOwnProperty("display")?r.display:null;s.style.display=a==null||typeof a=="boolean"?"":(""+a).trim()}}catch(l){pt(t,t.return,l)}um(t,e);break;case 6:try{t.stateNode.nodeValue=e?"":t.memoizedProps,nt=!0}catch(l){pt(t,t.return,l)}break;case 18:try{var o=t.stateNode;e?my(o,!0):my(t.stateNode,!1)}catch(l){pt(t,t.return,l)}break;case 22:case 23:t.memoizedState===null&&cm(t,e);break;default:cm(t,e)}}function um(t,e){if(t.subtreeFlags&67108864)for(t=t.child;t!==null;){e:{var n=t,i=e;switch(n.tag){case 4:US(n,i);break e;case 22:n.memoizedState===null&&um(n,i);break e;default:um(n,i)}}t=t.sibling}}function IS(t){var e=t.alternate;e!==null&&(t.alternate=null,IS(e)),t.child=null,t.deletions=null,t.sibling=null,t.tag===5&&(e=t.stateNode,e!==null&&cf(e)),t.stateNode=null,t.return=null,t.dependencies=null,t.memoizedProps=null,t.memoizedState=null,t.pendingProps=null,t.stateNode=null,t.updateQueue=null}var Bt=null,Gn=!1;function Ui(t,e,n){for(n=n.child;n!==null;)BS(t,e,n),n=n.sibling}function BS(t,e,n){if(ii&&typeof ii.onCommitFiberUnmount=="function")try{ii.onCommitFiberUnmount(ql,n)}catch{}switch(n.tag){case 26:lt||pn(n,e),Ui(t,e,n),n.memoizedState?n.memoizedState.count--:n.stateNode&&!lt&&(n=n.stateNode,n.parentNode.removeChild(n));break;case 27:lt||pn(n,e),bl(n);var i=Bt,s=Gn;Ar(n.type)&&(Bt=n.stateNode,Gn=!1),Ui(t,e,n),_A(n.stateNode,n.type,n.memoizedProps),Bt=i,Gn=s;break;case 5:lt||pn(n,e),bl(n);case 6:if(n.tag===6&&bl(n),i=Bt,s=Gn,Bt=null,Ui(t,e,n),Bt=i,Gn=s,Bt!==null)if(Gn)try{(Bt.nodeType===9?Bt.body:Bt.nodeName==="HTML"?Bt.ownerDocument.body:Bt).removeChild(n.stateNode),nt=!0}catch(r){pt(n,e,r)}else try{Bt.removeChild(n.stateNode),nt=!0}catch(r){pt(n,e,r)}break;case 18:Bt!==null&&(Gn?(t=Bt,py(t.nodeType===9?t.body:t.nodeName==="HTML"?t.ownerDocument.body:t,n.stateNode),yo(t)):py(Bt,n.stateNode));break;case 4:i=Bt,s=Gn,Bt=n.stateNode.containerInfo,Gn=!0,Ui(t,e,n),Bt=i,Gn=s;break;case 0:case 11:case 14:case 15:yr(2,n,e),lt||yr(4,n,e),Ui(t,e,n);break;case 1:lt||(pn(n,e),i=n.stateNode,typeof i.componentWillUnmount=="function"&&MS(n,e,i)),Ui(t,e,n);break;case 21:Ui(t,e,n);break;case 22:lt=(i=lt)||n.memoizedState!==null,Ui(t,e,n),lt=i;break;case 30:pn(n,e),Ui(t,e,n);break;case 7:lt||pn(n,e),Ui(t,e,n);break;default:Ui(t,e,n)}}function NS(t,e){if(e.memoizedState===null&&(t=e.alternate,t!==null&&(t=t.memoizedState,t!==null))){t=t.dehydrated;try{yo(t)}catch(n){pt(e,e.return,n)}}}function PS(t,e){if(e.memoizedState===null&&(t=e.alternate,t!==null&&(t=t.memoizedState,t!==null&&(t=t.dehydrated,t!==null))))try{yo(t)}catch(n){pt(e,e.return,n)}}function $T(t){switch(t.tag){case 31:case 13:case 19:var e=t.stateNode;return e===null&&(e=t.stateNode=new $x),e;case 22:return t=t.stateNode,e=t._retryCache,e===null&&(e=t._retryCache=new $x),e;default:throw Error(J(435,t.tag))}}function ou(t,e){var n=$T(t);e.forEach(function(i){if(!n.has(i)){n.add(i);var s=fb.bind(null,t,i);i.then(s,s)}})}function Nn(t,e,n){var i=e.deletions;if(i!==null)for(var s=0;s<i.length;s++){var r=i[s],a=t,o=e,l=o;e:for(;l!==null;){switch(l.tag){case 27:if(Ar(l.type)){Bt=l.stateNode,Gn=!1;break e}break;case 5:Bt=l.stateNode,Gn=!1;break e;case 3:case 4:Bt=l.stateNode.containerInfo,Gn=!0;break e}l=l.return}if(Bt===null)throw Error(J(160));BS(a,o,r),Bt=null,Gn=!1,a=r.alternate,a!==null&&(a.return=null),r.return=null}if(e.subtreeFlags&13886)for(e=e.child;e!==null;)LS(e,t,n),e=e.sibling}var Bi=null;function LS(t,e,n){var i=t.alternate,s=t.flags;switch(t.tag){case 0:case 11:case 14:case 15:if(s&4&&(i=t.updateQueue,i=i!==null?i.events:null,i!==null))for(var r=0;r<i.length;r++){var a=i[r];a.ref.impl=a.nextImpl}Nn(e,t,n),Pn(t),s&4&&(yr(3,t,t.return),ec(3,t),yr(5,t,t.return));break;case 1:Nn(e,t,n),Pn(t),s&512&&(lt||i===null||pn(i,i.return)),s&64&&ln&&(t=t.updateQueue,t!==null&&(e=t.callbacks,e!==null&&(n=t.shared.hiddenCallbacks,t.shared.hiddenCallbacks=n===null?e:n.concat(e))));break;case 26:if(r=Bi,Nn(e,t,n),Pn(t),s&512&&(lt||i===null||pn(i,i.return)),s&4)if(s=i!==null?i.memoizedState:null,n=t.memoizedState,i===null)if(n===null)if(t.stateNode===null)if(ln)t.stateNode=cA(t.type,t.memoizedProps,e.containerInfo,t);else{e:{e=t.type,n=t.memoizedProps,s=r.ownerDocument||r;t:switch(e){case"title":i=s.getElementsByTagName("title")[0],(!i||i[Kl]||i[mn]||i.namespaceURI==="http://www.w3.org/2000/svg"||i.hasAttribute("itemprop"))&&(i=s.createElement(e),s.head.insertBefore(i,s.querySelector("head > title"))),yn(i,e,n),i[mn]=t,un(i),e=i;break e;case"link":if(r=My("link","href",s).get(e+(n.href||""))){for(a=0;a<r.length;a++)if(i=r[a],i.getAttribute("href")===(n.href==null||n.href===""?null:n.href)&&i.getAttribute("rel")===(n.rel==null?null:n.rel)&&i.getAttribute("title")===(n.title==null?null:n.title)&&i.getAttribute("crossorigin")===(n.crossOrigin==null?null:n.crossOrigin)){r.splice(a,1);break t}}i=s.createElement(e),yn(i,e,n),s.head.appendChild(i);break;case"meta":if(r=My("meta","content",s).get(e+(n.content||""))){for(a=0;a<r.length;a++)if(i=r[a],i.getAttribute("content")===(n.content==null?null:""+n.content)&&i.getAttribute("name")===(n.name==null?null:n.name)&&i.getAttribute("property")===(n.property==null?null:n.property)&&i.getAttribute("http-equiv")===(n.httpEquiv==null?null:n.httpEquiv)&&i.getAttribute("charset")===(n.charSet==null?null:n.charSet)){r.splice(a,1);break t}}i=s.createElement(e),yn(i,e,n),s.head.appendChild(i);break;default:throw Error(J(468,e))}i[mn]=t,un(i),e=i}t.stateNode=e}else ln||wm(r,t.type,t.stateNode);else t.stateNode=Ay(r,n,t.memoizedProps);else s!==n?(s===null?(e=i.stateNode,e===null||lt||e.parentNode.removeChild(e)):s.count--,n===null?ln||wm(r,t.type,t.stateNode):Ay(r,n,t.memoizedProps)):n===null&&t.stateNode!==null&&lp(t,t.memoizedProps,i.memoizedProps);break;case 27:Nn(e,t,n),Pn(t),s&512&&(lt||i===null||pn(i,i.return)),i!==null&&s&4&&lp(t,t.memoizedProps,i.memoizedProps);break;case 5:if(r=Ki,Ki=!1,Nn(e,t,n),Ki=r,Pn(t),s&512&&(lt||i===null||pn(i,i.return)),t.flags&32){e=t.stateNode;try{lo(e,""),nt=!0}catch(h){pt(t,t.return,h)}}s&4&&t.stateNode!=null&&(e=t.memoizedProps,lp(t,e,i!==null?i.memoizedProps:e)),s&1024&&(up=!0);break;case 6:if(Nn(e,t,n),Pn(t),s&4){if(t.stateNode===null)throw Error(J(162));e=t.memoizedProps,n=t.stateNode;try{n.nodeValue=e,nt=!0}catch(h){pt(t,t.return,h)}}break;case 3:if(nt=!1,Cu=null,r=Bi,Bi=Vl(e.containerInfo),Nn(e,t,n),Bi=r,Pn(t),s&4&&i!==null&&i.memoizedState.isDehydrated)try{yo(e.containerInfo)}catch(h){pt(t,t.return,h)}up&&(up=!1,OS(t)),nt=!1;break;case 4:s=Ki,Ki=ln,i=rx(),r=Bi,Bi=Vl(t.stateNode.containerInfo),Nn(e,t,n),Pn(t),Bi=r,nt&&gl&&(ju=!0),nt=i,Ki=s;break;case 12:Nn(e,t,n),Pn(t);break;case 31:Nn(e,t,n),Pn(t),s&4&&(e=t.updateQueue,e!==null&&(t.updateQueue=null,ou(t,e)));break;case 13:Nn(e,t,n),Pn(t),t.child.flags&8192&&t.memoizedState!==null!=(i!==null&&i.memoizedState!==null)&&(Af=ni()),s&4&&(e=t.updateQueue,e!==null&&(t.updateQueue=null,ou(t,e)));break;case 22:r=t.memoizedState!==null,a=i!==null&&i.memoizedState!==null;var o=ln,l=lt,c=Ki;ln=o||r,Ki=c||r,lt=l||a,Nn(e,t,n),lt=l,Ki=c,ln=o,Pn(t),s&8192&&(e=t.stateNode,e._visibility=r?e._visibility&-2:e._visibility|1,!r||i===null||a||ln||lt||(e=a||lt,n=ln,i=lt,ln=r||ln,lt=e,Ks(t,2),ln=n,lt=i),!r&&Ki||cm(t,r)),s&4&&(e=t.updateQueue,e!==null&&(n=e.retryQueue,n!==null&&(e.retryQueue=null,ou(t,n))));break;case 19:Nn(e,t,n),Pn(t),s&4&&(e=t.updateQueue,e!==null&&(t.updateQueue=null,ou(t,e)));break;case 30:s&512&&(lt||i===null||pn(i,i.return)),s=rx(),r=gl,a=(n&335544064)===n,o=t.memoizedProps,gl=a&&Bs(o.default,o.update)!=="none",Nn(e,t,n),Pn(t),a&&i!==null&&nt&&(t.flags|=4),gl=r,nt=s;break;case 21:break;case 7:s&512&&(lt||i===null||pn(i,i.return)),i&&i.stateNode!==null&&(i.stateNode._fragmentFiber=t);default:Nn(e,t,n),Pn(t)}}function Pn(t){var e=t.flags;if(e&2){try{for(var n,i=t.return;i!==null;){if(ES(i)){n=i;break}i=i.return}i=null;for(var s=t.return;s!==null;){if(dg(s)){var r=s.stateNode;i===null?i=[r]:i.push(r)}if(hg(s))break;s=s.return}var a=i;if(n==null)throw Error(J(160));switch(n.tag){case 27:var o=n.stateNode,l=cp(t);Ku(t,l,o,a);break;case 5:var c=n.stateNode;n.flags&32&&(lo(c,""),n.flags&=-33);var h=cp(t);Ku(t,h,c,a);break;case 3:case 4:var p=n.stateNode.containerInfo,u=cp(t);im(t,u,p,a);break;default:throw Error(J(161))}}catch(d){pt(t,t.return,d)}t.flags&=-3}e&4096&&(t.flags&=-4097)}function OS(t){if(t.subtreeFlags&1024)for(t=t.child;t!==null;){var e=t;OS(e),e.tag===5&&e.flags&1024&&(e=e.stateNode,xo=!0,e.reset(),xo=!1),t=t.sibling}}function Na(t,e){if(e.subtreeFlags&9270)for(e=e.child;e!==null;)FS(e,t),e=e.sibling;else RS(e,!1)}function FS(t,e){var n=t.alternate;if(n===null)sm(t,!1);else switch(t.tag){case 3:if(lm=Ji=!1,jx(),Na(e,t),!Ji&&!ju){if(t=es,t!==null)for(var i=0;i<t.length;i+=3){n=t[i];var s=t[i+1];fA(n,t[i+2]),n=n.ownerDocument.documentElement,n!==null&&n.animate({opacity:[0,0],pointerEvents:["none","none"]},{duration:0,fill:"forwards",pseudoElement:"::view-transition-group("+s+")"})}t=e.containerInfo,t=t.nodeType===9?t.documentElement:t.ownerDocument.documentElement,t!==null&&t.style.viewTransitionName===""&&(t.style.viewTransitionName="none",t.animate({opacity:[0,0],pointerEvents:["none","none"]},{duration:0,fill:"forwards",pseudoElement:"::view-transition-group(root)"}),t.animate({width:[0,0],height:[0,0]},{duration:0,fill:"forwards",pseudoElement:"::view-transition"})),lm=!0}es=null;break;case 5:Na(e,t);break;case 4:i=Ji,Ji=!1,Na(e,t),Ji&&(ju=!0),Ji=i;break;case 22:t.memoizedState===null&&(n.memoizedState!==null?sm(t,!1):Na(e,t));break;case 30:i=Ji,s=jx(),Ji=!1,Na(e,t),Ji&&(t.flags|=4);var r=t.memoizedProps,a=t.stateNode;e=ws(r,a),a=ws(n.memoizedProps,a);var o=Bs(r.default,r.update);o==="none"?e=!1:(r=n.memoizedState,n.memoizedState=null,n=t.child,kn=0,e=pg(t,n,e,a,o,r,!0),kn!==(r===null?0:r.length)&&(t.flags|=32)),(t.flags&4)!==0&&e?(fo(t,t.memoizedProps.onUpdate),es=s):s!==null&&(s.push.apply(s,es),es=s),Ji=(t.flags&32)!==0?!0:i;break;default:Na(e,t)}}function ji(t,e){if(e.subtreeFlags&8772)for(e=e.child;e!==null;)DS(t,e.alternate,e),e=e.sibling}function Ks(t,e){for(t=t.child;t!==null;){var n=t,i=e;switch(n.tag){case 0:case 11:case 14:case 15:yr(4,n,n.return),Ks(n,i);break;case 1:pn(n,n.return);var s=n.stateNode;typeof s.componentWillUnmount=="function"&&MS(n,n.return,s),Ks(n,i);break;case 27:(i&2)!==0&&_A(n.stateNode,n.type,n.memoizedProps);case 5:pn(n,n.return),n.tag!==5&&n.tag!==27||bl(n),Ks(n,i);break;case 6:bl(n);break;case 26:pn(n,n.return),s=n.stateNode,n.memoizedState!==null||s===null||lt||s.parentNode.removeChild(s),Ks(n,i);break;case 22:n.memoizedState===null&&Ks(n,i);break;case 30:pn(n,n.return),Ks(n,i);break;case 7:pn(n,n.return);default:Ks(n,i)}t=t.sibling}}function Ii(t,e,n){for(n=(e.subtreeFlags&8772)!==0?n:n&-2,e=e.child;e!==null;){var i=e.alternate,s=t,r=e,a=r.flags,o=(n&1)!==0;switch(r.tag){case 0:case 11:case 15:Ii(s,r,n),ec(4,r);break;case 1:if(Ii(s,r,n),i=r,s=i.stateNode,typeof s.componentDidMount=="function")try{s.componentDidMount()}catch(h){pt(i,i.return,h)}if(i=r,s=i.updateQueue,s!==null){var l=i.stateNode;try{var c=s.shared.hiddenCallbacks;if(c!==null)for(s.shared.hiddenCallbacks=null,s=0;s<c.length;s++)U_(c[s],l)}catch(h){pt(i,i.return,h)}}o&&a&64&&AS(r),$i(r,r.return);break;case 27:(n&2)!==0&&TS(r);case 5:r.tag!==5&&r.tag!==27||Kx(r),Ii(s,r,n),o&&i===null&&a&4&&nm(r),$i(r,r.return);break;case 6:Kx(r);break;case 26:l=r.stateNode,r.memoizedState!==null||l===null||ln||wm(Vl(l.ownerDocument),r.type,l),Ii(s,r,n),o&&i===null&&a&4&&nm(r),$i(r,r.return);break;case 12:Ii(s,r,n);break;case 31:Ii(s,r,n),o&&a&4&&NS(s,r);break;case 13:Ii(s,r,n),o&&a&4&&PS(s,r);break;case 22:r.memoizedState===null&&Ii(s,r,n),$i(r,r.return);break;case 30:Ii(s,r,n),$i(r,r.return);break;case 7:$i(r,r.return);default:Ii(s,r,n)}e=e.sibling}}function mg(t,e){var n=null;t!==null&&t.memoizedState!==null&&t.memoizedState.cachePool!==null&&(n=t.memoizedState.cachePool.pool),t=null,e.memoizedState!==null&&e.memoizedState.cachePool!==null&&(t=e.memoizedState.cachePool.pool),t!==n&&(t!=null&&t.refCount++,n!=null&&jl(n))}function gg(t,e){t=null,e.alternate!==null&&(t=e.alternate.memoizedState.cache),e=e.memoizedState.cache,e!==t&&(e.refCount++,t!=null&&jl(t))}function mi(t,e,n,i){var s=(n&335544064)===n;if(e.subtreeFlags&(s?10262:10256))for(e=e.child;e!==null;)zS(t,e,n,i),e=e.sibling;else s&&CS(e)}function zS(t,e,n,i){var s=(n&335544064)===n;s&&e.alternate===null&&e.return!==null&&e.return.alternate!==null&&Eu(e);var r=e.flags;switch(e.tag){case 0:case 11:case 15:mi(t,e,n,i),r&2048&&ec(9,e);break;case 1:mi(t,e,n,i);break;case 3:mi(t,e,n,i),s&&lm&&(t=t.containerInfo,t=t.nodeType===9?t.body:t.nodeName==="HTML"?t.ownerDocument.body:t,t.style.viewTransitionName==="root"&&(t.style.viewTransitionName=""),t=t.ownerDocument.documentElement,t!==null&&t.style.viewTransitionName==="none"&&(t.style.viewTransitionName="")),r&2048&&(r=null,e.alternate!==null&&(r=e.alternate.memoizedState.cache),e=e.memoizedState.cache,e!==r&&(e.refCount++,r!=null&&jl(r)));break;case 12:if(r&2048){mi(t,e,n,i),r=e.stateNode;try{var a=e.memoizedProps,o=a.id,l=a.onPostCommit;typeof l=="function"&&l(o,e.alternate===null?"mount":"update",r.passiveEffectDuration,-0)}catch(c){pt(e,e.return,c)}}else mi(t,e,n,i);break;case 31:mi(t,e,n,i);break;case 13:mi(t,e,n,i);break;case 23:break;case 22:a=e.stateNode,o=e.alternate,e.memoizedState!==null?(s&&o!==null&&o.memoizedState===null&&Eu(o),a._visibility&2?mi(t,e,n,i):wl(t,e)):(s&&o!==null&&o.memoizedState!==null&&Eu(e),a._visibility&2?mi(t,e,n,i):(a._visibility|=2,La(t,e,n,i,(e.subtreeFlags&10256)!==0||!1))),r&2048&&mg(o,e);break;case 24:mi(t,e,n,i),r&2048&&gg(e.alternate,e);break;case 30:s&&(r=e.alternate,r!==null&&(os(r.child,!0),os(e.child,!0))),mi(t,e,n,i);break;default:mi(t,e,n,i)}}function La(t,e,n,i,s){for(s=s&&((e.subtreeFlags&10256)!==0||!1),e=e.child;e!==null;){var r=t,a=e,o=n,l=i,c=a.flags;switch(a.tag){case 0:case 11:case 15:La(r,a,o,l,s),ec(8,a);break;case 23:break;case 22:var h=a.stateNode;a.memoizedState!==null?h._visibility&2?La(r,a,o,l,s):wl(r,a):(h._visibility|=2,La(r,a,o,l,s)),s&&c&2048&&mg(a.alternate,a);break;case 24:La(r,a,o,l,s),s&&c&2048&&gg(a.alternate,a);break;default:La(r,a,o,l,s)}e=e.sibling}}function wl(t,e){if(e.subtreeFlags&10256)for(e=e.child;e!==null;){var n=t,i=e,s=i.flags;switch(i.tag){case 22:wl(n,i),s&2048&&mg(i.alternate,i);break;case 24:wl(n,i),s&2048&&gg(i.alternate,i);break;default:wl(n,i)}e=e.sibling}}var Yr=8192;function kr(t,e,n){if(t.subtreeFlags&Yr)for(t=t.child;t!==null;)HS(t,e,n),t=t.sibling}function HS(t,e,n){switch(t.tag){case 26:kr(t,e,n),t.flags&Yr&&(t.memoizedState!==null?iw(n,Bi,t.memoizedState,t.memoizedProps):(t=t.stateNode,(e&335544128)===e&&Ty(n,t)));break;case 5:kr(t,e,n),t.flags&Yr&&(t=t.stateNode,(e&335544128)===e&&Ty(n,t));break;case 3:case 4:var i=Bi;Bi=Vl(t.stateNode.containerInfo),kr(t,e,n),Bi=i;break;case 22:t.memoizedState===null&&(i=t.alternate,i!==null&&i.memoizedState!==null?(i=Yr,Yr=16777216,kr(t,e,n),Yr=i):kr(t,e,n));break;case 30:if((t.flags&Yr)!==0&&(i=t.memoizedProps.name,i!=null&&i!=="auto")){var s=t.stateNode;s.paired=null,ei===null&&(ei=new Map),ei.set(i,s)}kr(t,e,n);break;default:kr(t,e,n)}}function GS(t){var e=t.alternate;if(e!==null&&(t=e.child,t!==null)){e.child=null;do e=t.sibling,t.sibling=null,t=e;while(t!==null)}}function cl(t){var e=t.deletions;if((t.flags&16)!==0){if(e!==null)for(var n=0;n<e.length;n++){var i=e[n];cn=i,kS(i,t)}GS(t)}if(t.subtreeFlags&10256)for(t=t.child;t!==null;)VS(t),t=t.sibling}function VS(t){switch(t.tag){case 0:case 11:case 15:cl(t),t.flags&2048&&yr(9,t,t.return);break;case 3:cl(t);break;case 12:cl(t);break;case 22:var e=t.stateNode;t.memoizedState!==null&&e._visibility&2&&(t.return===null||t.return.tag!==13)?(e._visibility&=-3,Tu(t)):cl(t);break;default:cl(t)}}function Tu(t){var e=t.deletions;if((t.flags&16)!==0){if(e!==null)for(var n=0;n<e.length;n++){var i=e[n];cn=i,kS(i,t)}GS(t)}for(t=t.child;t!==null;){switch(e=t,e.tag){case 0:case 11:case 15:yr(8,e,e.return),Tu(e);break;case 22:n=e.stateNode,n._visibility&2&&(n._visibility&=-3,Tu(e));break;default:Tu(e)}t=t.sibling}}function kS(t,e){for(;cn!==null;){var n=cn;switch(n.tag){case 0:case 11:case 15:yr(8,n,e);break;case 23:case 22:if(n.memoizedState!==null&&n.memoizedState.cachePool!==null){var i=n.memoizedState.cachePool.pool;i!=null&&i.refCount++}break;case 24:jl(n.memoizedState.cache)}if(i=n.child,i!==null)i.return=n,cn=i;else e:for(n=t;cn!==null;){i=cn;var s=i.sibling,r=i.return;if(IS(i),i===n){cn=null;break e}if(s!==null){s.return=r,cn=s;break e}cn=r}}}var eb={getCacheForType:function(t){var e=gn(Zt),n=e.data.get(t);return n===void 0&&(n=t(),e.data.set(t,n)),n},cacheSignal:function(){return gn(Zt).controller.signal}},tb=typeof WeakMap=="function"?WeakMap:Map,st=0,xt=null,Xe=null,Ze=0,ht=0,Jn=null,ir=!1,Eo=!1,vg=!1,Us=0,Ht=0,_r=0,$r=0,$u=0,ti=0,uo=0,Cl=null,Vn=null,fm=!1,Af=0,WS=0,ef=1/0,tf=null,hr=null,Lt=0,Pi=null,oa=null,as=0,hm=0,dm=null,XS=null,io=null,so=null,ro=null,Rl=0,bu=null;function ri(){return(st&2)!==0&&Ze!==0?Ze&-Ze:Ue.T!==null?yg():Zy()}function YS(){if(ti===0)if((Ze&536870912)===0||Ge){var t=Qc;Qc<<=1,(Qc&3932160)===0&&(Qc=262144),ti=t}else ti=536870912;return t=_n.current,t!==null&&(t.flags|=32),ti}function fo(t,e){if(e!=null){var n=t.stateNode,i=n.ref;i===null&&(i=n.ref=dA(ws(t.memoizedProps,n))),so===null&&(so=[]),so.push(e.bind(null,i))}}function Xn(t,e,n){(t===xt&&(ht===2||ht===9)||t.cancelPendingCommit!==null)&&(ho(t,0),sr(t,Ze,ti,!1)),Zl(t,n),((st&2)===0||t!==xt)&&(t===xt&&((st&2)===0&&($r|=n),Ht===4&&sr(t,Ze,ti,!1)),cs(t))}function qS(t,e,n){if((st&6)!==0)throw Error(J(327));var i=!n&&(e&127)===0&&(e&t.expiredLanes)===0||Ql(t,e),s=i?sb(t,e):fp(t,e,!0),r=i;do{if(s===0){Eo&&!i&&sr(t,e,0,!1);break}else{if(n=t.current.alternate,r&&!nb(n)){s=fp(t,e,!1),r=!1;continue}if(s===2){if(r=e,t.errorRecoveryDisabledLanes&r)var a=0;else a=t.pendingLanes&-536870913,a=a!==0?a:a&536870912?536870912:0;if(a!==0){e=a;e:{var o=t;s=Cl;var l=o.current.memoizedState.isDehydrated;if(l&&(ho(o,a).flags|=256),a=fp(o,a,!1),a!==2&&a!==6){if(vg&&!l){o.errorRecoveryDisabledLanes|=r,$r|=r,s=4;break e}r=Vn,Vn=s,r!==null&&(Vn===null?Vn=r:Vn.push.apply(Vn,r))}s=a}if(r=!1,s!==2)continue}}if(s===1){ho(t,0),sr(t,e,0,!0);break}e:{switch(i=t,r=s,r){case 0:case 1:throw Error(J(345));case 4:if((e&4194048)!==e&&(e&62914560)!==e)break;case 6:sr(i,e,ti,!ir);break e;case 2:Vn=null;break;case 3:case 5:break;default:throw Error(J(329))}if((e&62914560)===e&&(s=Af+300-ni(),10<s)){if(sr(i,e,ti,!ir),lf(i,0,!0)!==0)break e;as=e,i.timeoutHandle=Sg(ey.bind(null,i,n,Vn,tf,fm,e,ti,$r,uo,ir,r,"Throttled",-0,0),s);break e}ey(i,n,Vn,tf,fm,e,ti,$r,uo,ir,r,null,-0,0)}}break}while(!0);cs(t)}function ey(t,e,n,i,s,r,a,o,l,c,h,p,u,d){t.timeoutHandle=-1;var v=e.subtreeFlags,M=(r&335544064)===r;if(p=null,(M||v&8192||(v&16785408)===16785408)&&(p={stylesheets:null,count:0,imgCount:0,imgBytes:0,suspenseyImages:[],waitingForImages:!0,waitingForViewTransition:!1,unsuspend:ns},ei=null,HS(e,r,p),M&&(v=p,M=t.containerInfo,M=(M.nodeType===9?M:M.ownerDocument).__reactViewTransition,M!=null&&(v.count++,v.waitingForViewTransition=!0,v=kl.bind(v),M.finished.then(v,v))),v=(r&62914560)===r?Af-ni():(r&4194048)===r?WS-ni():0,v=sw(p,v),v!==null)){as=r,t.cancelPendingCommit=v(ny.bind(null,t,e,r,n,i,s,a,o,l,c,h,p,null,u,d)),sr(t,r,a,!c);return}ny(t,e,r,n,i,s,a,o,l,c,h,p)}function nb(t){for(var e=t;;){var n=e.tag;if((n===0||n===11||n===15)&&e.flags&16384&&(n=e.updateQueue,n!==null&&(n=n.stores,n!==null)))for(var i=0;i<n.length;i++){var s=n[i],r=s.getSnapshot;s=s.value;try{if(!ai(r(),s))return!1}catch{return!1}}if(n=e.child,e.subtreeFlags&16384&&n!==null)n.return=e,e=n;else{if(e===t)break;for(;e.sibling===null;){if(e.return===null||e.return===t)return!0;e=e.return}e.sibling.return=e.return,e=e.sibling}}return!0}function sr(t,e,n,i){e=Wy(t,e),e&=~$u,e&=~$r,t.suspendedLanes|=e,t.pingedLanes&=~e,i&&(t.warmLanes|=e),i=t.expirationTimes;for(var s=e;0<s;){var r=31-si(s),a=1<<r;i[r]=-1,s&=~a}n!==0&&Yy(t,n,e)}function Mf(){return(st&6)===0?(tc(0,!1),!1):!0}function xg(){if(Xe!==null){if(ht===0)var t=Xe.return;else t=Xe,Ms=ha=null,tg(t),eo=null,Ll=0,t=Xe;for(;t!==null;)SS(t.alternate,t),t=t.return;Xe=null}}function ho(t,e){var n=t.timeoutHandle;return n!==-1&&(t.timeoutHandle=-1,Eb(n)),n=t.cancelPendingCommit,n!==null&&(t.cancelPendingCommit=null,n()),as=0,xg(),xt=t,Xe=n=Es(t.current,null),Ze=e,ht=0,Jn=null,ir=!1,Eo=Ql(t,e),vg=!1,uo=ti=$u=$r=_r=Ht=0,Vn=Cl=null,fm=!1,Us=Wy(t,e),df(),n}function QS(t,e){Oe=null,Ue.H=qu,e===Ao||e===gf?(e=wx(),ht=3):e===qm?(e=wx(),ht=4):ht=e===cg?8:e!==null&&typeof e=="object"&&typeof e.then=="function"?6:1,Jn=e,Xe===null&&(Ht=1,Qu(t,_i(e,t.current)))}function ZS(){var t=_n.current;return t===null?!0:(Ze&4194048)===Ze?Tn===null:(Ze&62914560)===Ze||(Ze&536870912)!==0?t===Tn:!1}function KS(){var t=Ue.H;return Ue.H=qu,t===null?qu:t}function JS(){var t=Ue.A;return Ue.A=eb,t}function nf(){Ht=4,ir||(Ze&4194048)!==Ze&&_n.current!==null||(Eo=!0),(_r&134217727)===0&&($r&134217727)===0||xt===null||sr(xt,Ze,ti,!1)}function fp(t,e,n){var i=st;st|=2;var s=KS(),r=JS();(xt!==t||Ze!==e)&&(tf=null,ho(t,e)),e=!1;var a=Ht;e:do try{if(ht!==0&&Xe!==null){var o=Xe,l=Jn;switch(ht){case 8:xg(),a=6;break e;case 3:case 2:case 9:case 6:_n.current===null&&(e=!0);var c=ht;if(ht=0,Jn=null,Za(t,o,l,c),n&&Eo){a=0;break e}break;default:c=ht,ht=0,Jn=null,Za(t,o,l,c)}}ib(),a=Ht;break}catch(h){QS(t,h)}while(!0);return e&&t.shellSuspendCounter++,Ms=ha=null,st=i,Ue.H=s,Ue.A=r,Xe===null&&(xt=null,Ze=0,df()),a}function ib(){for(;Xe!==null;)jS(Xe)}function sb(t,e){var n=st;st|=2;var i=KS(),s=JS();xt!==t||Ze!==e?(tf=null,ef=ni()+500,ho(t,e)):Eo=Ql(t,e);e:do try{if(ht!==0&&Xe!==null){e=Xe;var r=Jn;t:switch(ht){case 1:ht=0,Jn=null,Za(t,e,r,1);break;case 2:case 9:if(bx(r)){ht=0,Jn=null,ty(e);break}e=function(){ht!==2&&ht!==9||xt!==t||(ht=7),cs(t)},r.then(e,e);break e;case 3:ht=7;break e;case 4:ht=5;break e;case 7:bx(r)?(ht=0,Jn=null,ty(e)):(ht=0,Jn=null,Za(t,e,r,7));break;case 5:var a=null;switch(Xe.tag){case 26:a=Xe.memoizedState;case 5:case 27:var o=Xe;if(a?MA(a):o.stateNode.complete){ht=0,Jn=null;var l=o.sibling;if(l!==null)Xe=l;else{var c=o.return;c!==null?(Xe=c,Ef(c)):Xe=null}break t}}ht=0,Jn=null,Za(t,e,r,5);break;case 6:ht=0,Jn=null,Za(t,e,r,6);break;case 8:xg(),Ht=6;break e;default:throw Error(J(462))}}rb();break}catch(h){QS(t,h)}while(!0);return Ms=ha=null,Ue.H=i,Ue.A=s,st=n,Xe!==null?0:(xt=null,Ze=0,df(),Ht)}function rb(){for(;Xe!==null&&!AE();)jS(Xe)}function jS(t){var e=_S(t.alternate,t,Us);t.memoizedProps=t.pendingProps,e===null?Ef(t):Xe=e}function ty(t){var e=t,n=e.alternate;switch(e.tag){case 15:case 0:e=kx(n,e,e.pendingProps,e.type,void 0,Ze);break;case 11:e=kx(n,e,e.pendingProps,e.type.render,e.ref,Ze);break;case 5:tg(e);var i=e;i===fn&&(Ge?(Hu(i),i.tag===5&&i.stateNode!=null&&(Tt=i.stateNode)):(Hu(i),Ge=!0));default:SS(n,e),e=Xe=A_(e,Us),e=_S(n,e,Us)}t.memoizedProps=t.pendingProps,e===null?Ef(t):Xe=e}function Za(t,e,n,i){Ms=ha=null,tg(e),eo=null,Ll=0;var s=e.return;try{if(YT(t,s,e,n,Ze)){Ht=1,Qu(t,_i(n,t.current)),Xe=null;return}}catch(r){if(s!==null)throw Xe=s,r;Ht=1,Qu(t,_i(n,t.current)),Xe=null;return}e.flags&32768?(Ge||i===1?t=!0:Eo||(Ze&536870912)!==0?t=!1:(ir=t=!0,(i===2||i===9||i===3||i===6)&&(i=_n.current,i!==null&&i.tag===13&&(i.flags|=16384))),$S(e,t)):Ef(e)}function Ef(t){var e=t;do{if((e.flags&32768)!==0){$S(e,ir);return}t=e.return;var n=KT(e.alternate,e,Us);if(n!==null){Xe=n;return}if(e=e.sibling,e!==null){Xe=e;return}Xe=e=t}while(e!==null);Ht===0&&(Ht=5)}function $S(t,e){do{var n=JT(t.alternate,t);if(n!==null){n.flags&=32767,Xe=n;return}if(n=t.return,n!==null&&(n.flags|=32768,n.subtreeFlags=0,n.deletions=null),!e&&(t=t.sibling,t!==null)){Xe=t;return}Xe=t=n}while(t!==null);Ht=6,Xe=null}function ny(t,e,n,i,s,r,a,o,l,c,h,p){t.cancelPendingCommit=null;do Tf();while(Lt!==0);if((st&6)!==0)throw Error(J(327));if(e!==null){if(e===t.current)throw Error(J(177));t===xt&&(Xe=xt=null,Ze=0),oa=e,Pi=t,as=n,dm=s,XS=i,ab(t,e,n,a,o,l,p)}}function ab(t,e,n,i,s,r,a){var o=e.lanes|e.childLanes;if(hm=o,o|=Gm,IE(t,n,o,i,s,r),so=null,(n&335544064)===n?(ro=NT(t),i=10262):(ro=null,i=10256),(e.subtreeFlags&i)!==0||(e.flags&i)!==0?(t.callbackNode=null,t.callbackPriority=0,hb(Pu,function(){return vm(),null})):(t.callbackNode=null,t.callbackPriority=0),Ju=!1,i=(e.flags&13878)!==0,(e.subtreeFlags&13878)!==0||i){i=Ue.T,Ue.T=null,s=rt.p,rt.p=2,r=st,st|=4;try{jT(t,e,n)}finally{st=r,rt.p=s,Ue.T=i}}Lt=1,Ju?io=Db(a,t.containerInfo,ro,pm,mm,lb,gm,vm,ob,null,null):(pm(),mm(),gm())}function ob(t){if(Lt!==0){var e=Pi.onRecoverableError;e(t,{componentStack:null})}}function lb(){Lt===3&&(Lt=0,FS(oa,Pi),Lt=4)}function pm(){if(Lt===1){Lt=0;var t=Pi,e=oa,n=as,i=(e.flags&13878)!==0;if((e.subtreeFlags&13878)!==0||i){i=Ue.T,Ue.T=null;var s=rt.p;rt.p=2;var r=st;st|=4;try{gl=ju=!1,LS(e,t,n),n=Sm;var a=p_(t.containerInfo),o=n.focusedElem,l=n.selectionRange;if(a!==o&&o&&o.ownerDocument&&d_(o.ownerDocument.documentElement,o)){if(l!==null&&Hm(o)){var c=l.start,h=l.end;if(h===void 0&&(h=c),"selectionStart"in o)o.selectionStart=c,o.selectionEnd=Math.min(h,o.value.length);else{var p=o.ownerDocument||document,u=p&&p.defaultView||window;if(u.getSelection){var d=u.getSelection(),v=o.textContent.length,M=Math.min(l.start,v),m=l.end===void 0?M:Math.min(l.end,v);!d.extend&&M>m&&(a=m,m=M,M=a);var f=xx(o,M),g=xx(o,m);if(f&&g&&(d.rangeCount!==1||d.anchorNode!==f.node||d.anchorOffset!==f.offset||d.focusNode!==g.node||d.focusOffset!==g.offset)){var S=p.createRange();S.setStart(f.node,f.offset),d.removeAllRanges(),M>m?(d.addRange(S),d.extend(g.node,g.offset)):(S.setEnd(g.node,g.offset),d.addRange(S))}}}}for(p=[],d=o;d=d.parentNode;)d.nodeType===1&&p.push({element:d,left:d.scrollLeft,top:d.scrollTop});for(typeof o.focus=="function"&&o.focus(),o=0;o<p.length;o++){var _=p[o];_.element.scrollLeft=_.left,_.element.scrollTop=_.top}}xo=!!_m,Sm=_m=null}finally{st=r,rt.p=s,Ue.T=i}}t.current=e,Lt=2}}function mm(){if(Lt===2){Lt=0;var t=Pi,e=oa,n=(e.flags&8772)!==0;if((e.subtreeFlags&8772)!==0||n){n=Ue.T,Ue.T=null;var i=rt.p;rt.p=2;var s=st;st|=4;try{DS(t,e.alternate,e)}finally{st=s,rt.p=i,Ue.T=n}}Lt=3}}function gm(){if(Lt===4||Lt===3){Lt=0;var t=io;io=null,ME();var e=Pi,n=oa,i=as,s=XS,r=(i&335544064)===i?10262:10256;if((n.subtreeFlags&r)!==0||(n.flags&r)!==0?Lt=5:(Lt=0,oa=Pi=null,eA(e,e.pendingLanes)),r=e.pendingLanes,r===0&&(hr=null),Nm(i),n=n.stateNode,ii&&typeof ii.onCommitFiberRoot=="function")try{ii.onCommitFiberRoot(ql,n,void 0,(n.current.flags&128)===128)}catch{}if(s!==null){n=Ue.T,r=rt.p,rt.p=2,Ue.T=null;try{for(var a=e.onRecoverableError,o=0;o<s.length;o++){var l=s[o];a(l.value,{componentStack:l.stack})}}finally{Ue.T=n,rt.p=r}}if(s=so,a=ro,ro=null,s!==null&&(so=null,a===null&&(a=[]),t!==null))for(l=0;l<s.length;l++)n=(0,s[l])(a),n!==void 0&&t.finished.finally(n);(as&3)!==0&&Tf(),cs(e),r=e.pendingLanes,(i&261930)!==0&&(r&42)!==0?e===bu?Rl++:(Rl=0,bu=e):(Rl=0,bu=null),tc(0,!1)}}function eA(t,e){(t.pooledCacheLanes&=e)===0&&(e=t.pooledCache,e!=null&&(t.pooledCache=null,jl(e)))}function Tf(){return io!==null&&(io.skipTransition(),io=null),pm(),mm(),gm(),vm()}function vm(){if(Lt!==5)return!1;var t=Pi,e=hm;hm=0;var n=Nm(as),i=Ue.T,s=rt.p;try{rt.p=32>n?32:n,Ue.T=null,n=dm,dm=null;var r=Pi,a=as;if(Lt=0,oa=Pi=null,as=0,(st&6)!==0)throw Error(J(331));var o=st;if(st|=4,VS(r.current),zS(r,r.current,a,n),st=o,tc(0,!1),ii&&typeof ii.onPostCommitFiberRoot=="function")try{ii.onPostCommitFiberRoot(ql,r)}catch{}return!0}finally{rt.p=s,Ue.T=i,eA(t,e)}}function iy(t,e,n){e=_i(n,e),e=Kp(t.stateNode,e,2),t=cr(t,e,2),t!==null&&(Zl(t,2),cs(t))}function pt(t,e,n){if(t.tag===3)iy(t,t,n);else for(;e!==null;){if(e.tag===3){iy(e,t,n);break}else if(e.tag===1){var i=e.stateNode;if(typeof e.type.getDerivedStateFromError=="function"||typeof i.componentDidCatch=="function"&&(hr===null||!hr.has(i))){t=_i(n,t),n=mS(2),i=cr(e,n,2),i!==null&&(gS(n,i,e,t),Zl(i,2),cs(i));break}}e=e.return}}function hp(t,e,n){var i=t.pingCache;if(i===null){i=t.pingCache=new tb;var s=new Set;i.set(e,s)}else s=i.get(e),s===void 0&&(s=new Set,i.set(e,s));s.has(n)||(vg=!0,s.add(n),t=cb.bind(null,t,e,n),e.then(t,t))}function cb(t,e,n){var i=t.pingCache;i!==null&&i.delete(e),t.pingedLanes|=t.suspendedLanes&n,t.warmLanes&=~n,xt===t&&(Ze&n)===n&&((Ht===4||Ht===3&&(Ze&62914560)===Ze&&300>ni()-Af)&&(st&2)===0?ho(t,0):$u|=n,uo===Ze&&(uo=0)),cs(t)}function tA(t,e){e===0&&(e=Xy()),t=fa(t,e),t!==null&&(Zl(t,e),cs(t))}function ub(t){var e=t.memoizedState,n=0;e!==null&&(n=e.retryLane),tA(t,n)}function fb(t,e){var n=0;switch(t.tag){case 31:case 13:var i=t.stateNode,s=t.memoizedState;s!==null&&(n=s.retryLane);break;case 19:i=t.stateNode;break;case 22:i=t.stateNode._retryCache;break;default:throw Error(J(314))}i!==null&&i.delete(e),tA(t,n)}function hb(t,e){return Im(t,e)}var po=null,Oa=null,xm=!1,sf=!1,dp=!1,rr=0;function cs(t){t!==Oa&&t.next===null&&(Oa===null?po=Oa=t:Oa=Oa.next=t),sf=!0,xm||(xm=!0,pb())}function tc(t,e){if(!dp&&sf){dp=!0;do for(var n=!1,i=po;i!==null;){if(!e)if(t!==0){var s=i.pendingLanes;if(s===0)var r=0;else{var a=i.suspendedLanes,o=i.pingedLanes;r=(1<<31-si(42|t)+1)-1,r&=s&~(a&~o),r=r&201326741?r&201326741|1:r?r|2:0}r!==0&&(n=!0,sy(i,r))}else r=Ze,r=lf(i,i===xt?r:0,i.cancelPendingCommit!==null||i.timeoutHandle!==-1),(r&3)===0||Ql(i,r)||(n=!0,sy(i,r));i=i.next}while(n);dp=!1}}function db(){nA()}function nA(){sf=xm=!1;var t=0;rr!==0&&Mb()&&(t=rr);for(var e=ni(),n=null,i=po;i!==null;){var s=i.next,r=iA(i,e);r===0?(i.next=null,n===null?po=s:n.next=s,s===null&&(Oa=n)):(n=i,(t!==0||(r&3)!==0)&&(sf=!0)),i=s}Lt!==0&&Lt!==5||tc(t,!1),rr!==0&&(rr=0)}function iA(t,e){for(var n=t.suspendedLanes,i=t.pingedLanes,s=t.expirationTimes,r=t.pendingLanes&-62914561;0<r;){var a=31-si(r),o=1<<a,l=s[a];l===-1?((o&n)===0||(o&i)!==0)&&(s[a]=UE(o,e)):l<=e&&(t.expiredLanes|=o),r&=~o}if(e=xt,n=Ze,n=lf(t,t===e?n:0,t.cancelPendingCommit!==null||t.timeoutHandle!==-1),i=t.callbackNode,n===0||t===e&&(ht===2||ht===9)||t.cancelPendingCommit!==null)return i!==null&&i!==null&&Xd(i),t.callbackNode=null,t.callbackPriority=0;if((n&3)===0||Ql(t,n)){if(e=n&-n,e===t.callbackPriority)return e;switch(i!==null&&Xd(i),Nm(n)){case 2:case 8:n=Vy;break;case 32:n=Pu;break;case 268435456:n=ky;break;default:n=Pu}return i=sA.bind(null,t),n=Im(n,i),t.callbackPriority=e,t.callbackNode=n,e}return i!==null&&i!==null&&Xd(i),t.callbackPriority=2,t.callbackNode=null,2}function sA(t,e){if(Lt!==0&&Lt!==5)return t.callbackNode=null,t.callbackPriority=0,null;var n=t.callbackNode;if(Tf()&&t.callbackNode!==n)return null;var i=Ze;return i=lf(t,t===xt?i:0,t.cancelPendingCommit!==null||t.timeoutHandle!==-1),i===0?null:(qS(t,i,e),iA(t,ni()),t.callbackNode!=null&&t.callbackNode===n?sA.bind(null,t):null)}function sy(t,e){if(Tf())return null;qS(t,e,!0)}function pb(){Tb(function(){(st&6)!==0?Im(Gy,db):nA()})}function yg(){if(rr===0){var t=ia;t===0&&(t=qc,qc<<=1,(qc&261888)===0&&(qc=256)),rr=t}return rr}function ry(t){return t==null||typeof t=="symbol"||typeof t=="boolean"?null:typeof t=="function"?t:pu(t)}function mb(t,e,n,i,s){if(e==="submit"&&n&&n.stateNode===s){var r=ry((s[qn]||null).action),a=i.submitter;a&&(e=(e=a[qn]||null)?ry(e.formAction):a.getAttribute("formAction"),e!==null&&(r=e,a=null));var o=new uf("action","action",null,i,s);t.push({event:o,listeners:[{instance:null,listener:function(){if(i.defaultPrevented){if(rr!==0){var l=new FormData(s,a);Qp(n,{pending:!0,data:l,method:s.method,action:r},null,l)}}else typeof r=="function"&&(o.preventDefault(),l=new FormData(s,a),Qp(n,{pending:!0,data:l,method:s.method,action:r},r,l))},currentTarget:s}]})}}for(lu=0;lu<Op.length;lu++)cu=Op[lu],ay=cu.toLowerCase(),oy=cu[0].toUpperCase()+cu.slice(1),Li(ay,"on"+oy);var cu,ay,oy,lu;Li(g_,"onAnimationEnd");Li(v_,"onAnimationIteration");Li(x_,"onAnimationStart");Li("dblclick","onDoubleClick");Li("focusin","onFocus");Li("focusout","onBlur");Li(bT,"onTransitionRun");Li(wT,"onTransitionStart");Li(CT,"onTransitionCancel");Li(y_,"onTransitionEnd");oo("onMouseEnter",["mouseout","mouseover"]);oo("onMouseLeave",["mouseout","mouseover"]);oo("onPointerEnter",["pointerout","pointerover"]);oo("onPointerLeave",["pointerout","pointerover"]);ca("onChange","change click focusin focusout input keydown keyup selectionchange".split(" "));ca("onSelect","focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange".split(" "));ca("onBeforeInput",["compositionend","keypress","textInput","paste"]);ca("onCompositionEnd","compositionend focusout keydown keypress keyup mousedown".split(" "));ca("onCompositionStart","compositionstart focusout keydown keypress keyup mousedown".split(" "));ca("onCompositionUpdate","compositionupdate focusout keydown keypress keyup mousedown".split(" "));var zl="abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(" "),gb=new Set("beforetoggle cancel close invalid load scroll scrollend toggle".split(" ").concat(zl));function rA(t,e){e=(e&4)!==0;for(var n=0;n<t.length;n++){var i=t[n],s=i.event;i=i.listeners;e:{var r=void 0;if(e)for(var a=i.length-1;0<=a;a--){var o=i[a],l=o.instance,c=o.currentTarget;if(o=o.listener,l!==r&&s.isPropagationStopped())break e;r=o,s.currentTarget=c;try{r(s)}catch(h){Ou(h)}s.currentTarget=null,r=l}else for(a=0;a<i.length;a++){if(o=i[a],l=o.instance,c=o.currentTarget,o=o.listener,l!==r&&s.isPropagationStopped())break e;r=o,s.currentTarget=c;try{r(s)}catch(h){Ou(h)}s.currentTarget=null,r=l}}}}function We(t,e){var n=e[tx];n===void 0&&(n=e[tx]=new Set);var i=t+"__bubble";n.has(i)||(aA(e,t,2,!1),n.add(i))}function pp(t,e,n){var i=0;e&&(i|=4),aA(n,t,i,e)}var uu="_reactListening"+Math.random().toString(36).slice(2);function _g(t){if(!t[uu]){t[uu]=!0,Jy.forEach(function(n){n!=="selectionchange"&&(gb.has(n)||pp(n,!1,t),pp(n,!0,t))});var e=t.nodeType===9?t:t.ownerDocument;e===null||e[uu]||(e[uu]=!0,pp("selectionchange",!1,e))}}function aA(t,e,n,i){switch(DA(e)){case 2:var s=lw;break;case 8:s=cw;break;default:s=wg}n=s.bind(null,e,n,t),s=void 0,!Bp||e!=="touchstart"&&e!=="touchmove"&&e!=="wheel"||(s=!0),i?s!==void 0?t.addEventListener(e,n,{capture:!0,passive:s}):t.addEventListener(e,n,!0):s!==void 0?t.addEventListener(e,n,{passive:s}):t.addEventListener(e,n,!1)}function mp(t,e,n,i,s){var r=i;if((e&1)===0&&(e&2)===0&&i!==null)e:for(;;){if(i===null)return;var a=i.tag;if(a===3||a===4){var o=i.stateNode.containerInfo;if(o===s)break;if(a===4)for(a=i.return;a!==null;){var l=a.tag;if((l===3||l===4)&&a.stateNode.containerInfo===s)return;a=a.return}for(;o!==null;){if(a=qr(o),a===null)return;if(l=a.tag,l===5||l===6||l===26||l===27){i=r=a;continue e}o=o.parentNode}}i=i.return}r_(function(){var c=r,h=Lm(n),p=[];e:{var u=__.get(t);if(u!==void 0){var d=uf,v=t;switch(t){case"keypress":if(gu(n)===0)break e;case"keydown":case"keyup":d=nT;break;case"focusin":v="focus",d=Jd;break;case"focusout":v="blur",d=Jd;break;case"beforeblur":case"afterblur":d=Jd;break;case"click":if(n.button===2)break e;case"auxclick":case"dblclick":case"mousedown":case"mousemove":case"mouseup":case"mouseout":case"mouseover":case"contextmenu":d=cx;break;case"drag":case"dragend":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"dragstart":case"drop":d=WE;break;case"touchcancel":case"touchend":case"touchmove":case"touchstart":d=oT;break;case g_:case v_:case x_:d=qE;break;case y_:d=cT;break;case"scroll":case"scrollend":d=VE;break;case"wheel":d=fT;break;case"copy":case"cut":case"paste":d=ZE;break;case"gotpointercapture":case"lostpointercapture":case"pointercancel":case"pointerdown":case"pointermove":case"pointerout":case"pointerover":case"pointerup":d=fx;break;case"submit":d=rT;break;case"toggle":case"beforetoggle":d=dT}var M=(e&4)!==0,m=!M&&(t==="scroll"||t==="scrollend"),f=M?u!==null?u+"Capture":null:u;M=[];for(var g=c,S;g!==null;){var _=g;if(S=_.stateNode,_=_.tag,_!==5&&_!==26&&_!==27||S===null||f===null||(_=Ul(g,f),_!=null&&M.push(Hl(g,_,S))),m)break;g=g.return}0<M.length&&(u=new d(u,v,null,n,h),p.push({event:u,listeners:M}))}}if((e&7)===0){e:{if(d=t==="mouseover"||t==="pointerover",u=t==="mouseout"||t==="pointerout",d&&n!==Ip&&(v=n.relatedTarget||n.fromElement)&&(qr(v)||v[_o]))break e;(u||d)&&(v=h.window===h?h:(d=h.ownerDocument)?d.defaultView||d.parentWindow:window,u?(d=n.relatedTarget||n.toElement,u=c,d=d?qr(d):null,d!==null&&(m=Yl(d),M=d.tag,d!==m||M!==5&&M!==27&&M!==6)&&(d=null)):(u=null,d=c),u!==d&&(M=cx,_="onMouseLeave",f="onMouseEnter",g="mouse",(t==="pointerout"||t==="pointerover")&&(M=fx,_="onPointerLeave",f="onPointerEnter",g="pointer"),m=u==null?v:pl(u),S=d==null?v:pl(d),v=new M(_,g+"leave",u,n,h),v.target=m,v.relatedTarget=S,_=null,qr(h)===c&&(M=new M(f,g+"enter",d,n,h),M.target=S,M.relatedTarget=m,_=M),m=_,M=u&&d?_p(u,d,vb):null,u!==null&&ly(p,v,u,M,!1),d!==null&&m!==null&&ly(p,m,d,M,!0)))}e:{if(u=c?pl(c):window,d=u.nodeName&&u.nodeName.toLowerCase(),d==="select"||d==="input"&&u.type==="file")var E=mx;else if(px(u))if(f_)E=MT;else{E=ST;var T=_T}else d=u.nodeName,!d||d.toLowerCase()!=="input"||u.type!=="checkbox"&&u.type!=="radio"?c&&Pm(c.elementType)&&(E=mx):E=AT;if(E&&(E=E(t,c))){u_(p,E,n,h);break e}T&&T(t,u,c)}switch(T=c?pl(c):window,t){case"focusin":(px(T)||T.contentEditable==="true")&&(ka=T,Pp=c,yl=null);break;case"focusout":yl=Pp=ka=null;break;case"mousedown":Lp=!0;break;case"contextmenu":case"mouseup":case"dragend":Lp=!1,yx(p,n,h);break;case"selectionchange":if(TT)break;case"keydown":case"keyup":yx(p,n,h)}var C;if(zm)e:{switch(t){case"compositionstart":var y="onCompositionStart";break e;case"compositionend":y="onCompositionEnd";break e;case"compositionupdate":y="onCompositionUpdate";break e}y=void 0}else Va?l_(t,n)&&(y="onCompositionEnd"):t==="keydown"&&n.keyCode===229&&(y="onCompositionStart");y&&(o_&&n.locale!=="ko"&&(Va||y!=="onCompositionStart"?y==="onCompositionEnd"&&Va&&(C=a_()):(tr=h,Om="value"in tr?tr.value:tr.textContent,Va=!0)),T=rf(c,y),0<T.length&&(y=new ux(y,t,null,n,h),p.push({event:y,listeners:T}),C?y.data=C:(C=c_(n),C!==null&&(y.data=C)))),(C=mT?gT(t,n):vT(t,n))&&(y=rf(c,"onBeforeInput"),0<y.length&&(T=new ux("onBeforeInput","beforeinput",null,n,h),p.push({event:T,listeners:y}),T.data=C)),mb(p,t,c,n,h)}rA(p,e)})}function Hl(t,e,n){return{instance:t,listener:e,currentTarget:n}}function rf(t,e){for(var n=e+"Capture",i=[];t!==null;){var s=t,r=s.stateNode;if(s=s.tag,s!==5&&s!==26&&s!==27||r===null||(s=Ul(t,n),s!=null&&i.unshift(Hl(t,s,r)),s=Ul(t,e),s!=null&&i.push(Hl(t,s,r))),t.tag===3)return i;t=t.return}return[]}function vb(t){if(t===null)return null;do t=t.return;while(t&&t.tag!==5&&t.tag!==27);return t||null}function ly(t,e,n,i,s){for(var r=e._reactName,a=[];n!==null&&n!==i;){var o=n,l=o.alternate,c=o.stateNode;if(o=o.tag,l!==null&&l===i)break;o!==5&&o!==26&&o!==27||c===null||(l=c,s?(c=Ul(n,r),c!=null&&a.unshift(Hl(n,c,l))):s||(c=Ul(n,r),c!=null&&a.push(Hl(n,c,l)))),n=n.return}a.length!==0&&t.push({event:e,listeners:a})}var xb=/\r\n?/g,yb=/\u0000|\uFFFD/g;function cy(t){return(typeof t=="string"?t:""+t).replace(xb,`
`).replace(yb,"")}function oA(t,e){return e=cy(e),cy(t)===e}function dt(t,e,n,i,s,r){switch(n){case"children":if(typeof i=="string")e==="body"||e==="textarea"&&i===""||lo(t,i);else if(typeof i=="number"||typeof i=="bigint")e!=="body"&&lo(t,""+i);else return;break;case"className":Kc(t,"class",i);break;case"tabIndex":Kc(t,"tabindex",i);break;case"dir":case"role":case"viewBox":case"width":case"height":Kc(t,n,i);break;case"style":s_(t,i,r);return;case"data":if(e!=="object"){Kc(t,"data",i);break}case"src":case"href":if(i===""&&(e!=="a"||n!=="href")){t.removeAttribute(n);break}if(i==null||typeof i=="function"||typeof i=="symbol"||typeof i=="boolean"){t.removeAttribute(n);break}i=pu(i),t.setAttribute(n,i);break;case"action":case"formAction":if(typeof i=="function"){t.setAttribute(n,"javascript:throw new Error('A React form was unexpectedly submitted. If you called form.submit() manually, consider using form.requestSubmit() instead. If you\\'re trying to use event.stopPropagation() in a submit event handler, consider also calling event.preventDefault().')");break}else typeof r=="function"&&(n==="formAction"?(e!=="input"&&dt(t,e,"name",s.name,s,null),dt(t,e,"formEncType",s.formEncType,s,null),dt(t,e,"formMethod",s.formMethod,s,null),dt(t,e,"formTarget",s.formTarget,s,null)):(dt(t,e,"encType",s.encType,s,null),dt(t,e,"method",s.method,s,null),dt(t,e,"target",s.target,s,null)));if(i==null||typeof i=="symbol"||typeof i=="boolean"){t.removeAttribute(n);break}i=pu(i),t.setAttribute(n,i);break;case"onClick":i!=null&&(t.onclick=ns);return;case"onScroll":i!=null&&We("scroll",t);return;case"onScrollEnd":i!=null&&We("scrollend",t);return;case"dangerouslySetInnerHTML":if(i!=null){if(typeof i!="object"||!("__html"in i))throw Error(J(61));if(n=i.__html,n!=null){if(s.children!=null)throw Error(J(60));r?.__html!==n&&(t.innerHTML=n)}}break;case"multiple":t.multiple=i&&typeof i!="function"&&typeof i!="symbol";break;case"muted":t.muted=i&&typeof i!="function"&&typeof i!="symbol";break;case"suppressContentEditableWarning":case"suppressHydrationWarning":case"defaultValue":case"defaultChecked":case"innerHTML":case"ref":break;case"autoFocus":break;case"xlinkHref":if(i==null||typeof i=="function"||typeof i=="boolean"||typeof i=="symbol"){t.removeAttribute("xlink:href");break}n=pu(i),t.setAttributeNS("http://www.w3.org/1999/xlink","xlink:href",n);break;case"contentEditable":case"spellCheck":case"draggable":case"value":case"autoReverse":case"externalResourcesRequired":case"focusable":case"preserveAlpha":i!=null&&typeof i!="function"&&typeof i!="symbol"?t.setAttribute(n,i):t.removeAttribute(n);break;case"inert":case"allowFullScreen":case"async":case"autoPlay":case"controls":case"credentialless":case"default":case"defer":case"disabled":case"disablePictureInPicture":case"disableRemotePlayback":case"formNoValidate":case"hidden":case"loop":case"noModule":case"noValidate":case"open":case"playsInline":case"readOnly":case"required":case"reversed":case"scoped":case"seamless":case"itemScope":i&&typeof i!="function"&&typeof i!="symbol"?t.setAttribute(n,""):t.removeAttribute(n);break;case"capture":case"download":i===!0?t.setAttribute(n,""):i!==!1&&i!=null&&typeof i!="function"&&typeof i!="symbol"?t.setAttribute(n,i):t.removeAttribute(n);break;case"cols":case"rows":case"size":case"span":i!=null&&typeof i!="function"&&typeof i!="symbol"&&!isNaN(i)&&1<=i?t.setAttribute(n,i):t.removeAttribute(n);break;case"rowSpan":case"start":i==null||typeof i=="function"||typeof i=="symbol"||isNaN(i)?t.removeAttribute(n):t.setAttribute(n,i);break;case"popover":We("beforetoggle",t),We("toggle",t),du(t,"popover",i);break;case"xlinkActuate":_s(t,"http://www.w3.org/1999/xlink","xlink:actuate",i);break;case"xlinkArcrole":_s(t,"http://www.w3.org/1999/xlink","xlink:arcrole",i);break;case"xlinkRole":_s(t,"http://www.w3.org/1999/xlink","xlink:role",i);break;case"xlinkShow":_s(t,"http://www.w3.org/1999/xlink","xlink:show",i);break;case"xlinkTitle":_s(t,"http://www.w3.org/1999/xlink","xlink:title",i);break;case"xlinkType":_s(t,"http://www.w3.org/1999/xlink","xlink:type",i);break;case"xmlBase":_s(t,"http://www.w3.org/XML/1998/namespace","xml:base",i);break;case"xmlLang":_s(t,"http://www.w3.org/XML/1998/namespace","xml:lang",i);break;case"xmlSpace":_s(t,"http://www.w3.org/XML/1998/namespace","xml:space",i);break;case"is":du(t,"is",i);break;case"innerText":case"textContent":return;default:if(!(2<n.length)||n[0]!=="o"&&n[0]!=="O"||n[1]!=="n"&&n[1]!=="N")n=HE.get(n)||n,du(t,n,i);else return}nt=!0}function ym(t,e,n,i,s,r){switch(n){case"style":s_(t,i,r);return;case"dangerouslySetInnerHTML":if(i!=null){if(typeof i!="object"||!("__html"in i))throw Error(J(61));if(n=i.__html,n!=null){if(s.children!=null)throw Error(J(60));r?.__html!==n&&(t.innerHTML=n)}}break;case"children":if(typeof i=="string")lo(t,i);else if(typeof i=="number"||typeof i=="bigint")lo(t,""+i);else return;break;case"onScroll":i!=null&&We("scroll",t);return;case"onScrollEnd":i!=null&&We("scrollend",t);return;case"onClick":i!=null&&(t.onclick=ns);return;case"suppressContentEditableWarning":case"suppressHydrationWarning":case"innerHTML":case"ref":return;case"innerText":case"textContent":return;default:if(!jy.hasOwnProperty(n))e:{if(n[0]==="o"&&n[1]==="n"&&(s=n.endsWith("Capture"),r=n.slice(2,s?n.length-7:void 0),e=t[qn]||null,e=e!=null?e[n]:null,typeof e=="function"&&t.removeEventListener(r,e,s),typeof i=="function")){typeof e!="function"&&e!==null&&(n in t?t[n]=null:t.hasAttribute(n)&&t.removeAttribute(n)),t.addEventListener(r,i,s);break e}nt=!0,n in t?t[n]=i:i===!0?t.setAttribute(n,""):du(t,n,i)}return}nt=!0}function yn(t,e,n){switch(e){case"div":case"span":case"svg":case"path":case"a":case"g":case"p":case"li":break;case"img":We("error",t),We("load",t);var i=!1,s=!1,r;for(r in n)if(n.hasOwnProperty(r)){var a=n[r];if(a!=null)switch(r){case"src":i=!0;break;case"srcSet":s=!0;break;case"children":case"dangerouslySetInnerHTML":throw Error(J(137,e));default:dt(t,e,r,a,n,null)}}s&&dt(t,e,"srcSet",n.srcSet,n,null),i&&dt(t,e,"src",n.src,n,null);return;case"input":We("invalid",t);var o=r=a=s=null,l=null,c=null;for(i in n)if(n.hasOwnProperty(i)){var h=n[i];if(h!=null)switch(i){case"name":s=h;break;case"type":a=h;break;case"checked":l=h;break;case"defaultChecked":c=h;break;case"value":r=h;break;case"defaultValue":o=h;break;case"children":case"dangerouslySetInnerHTML":if(h!=null)throw Error(J(137,e));break;default:dt(t,e,i,h,n,null)}}t_(t,r,o,l,c,a,s,!1);return;case"select":We("invalid",t),i=a=r=null;for(s in n)if(n.hasOwnProperty(s)&&(o=n[s],o!=null))switch(s){case"value":r=o;break;case"defaultValue":a=o;break;case"multiple":i=o;default:dt(t,e,s,o,n,null)}e=r,n=a,t.multiple=!!i,e!=null?Ja(t,!!i,e,!1):n!=null&&Ja(t,!!i,n,!0);return;case"textarea":We("invalid",t),r=s=i=null;for(a in n)if(n.hasOwnProperty(a)&&(o=n[a],o!=null))switch(a){case"value":i=o;break;case"defaultValue":s=o;break;case"children":r=o;break;case"dangerouslySetInnerHTML":if(o!=null)throw Error(J(91));break;default:dt(t,e,a,o,n,null)}i_(t,i,s,r);return;case"option":for(l in n)n.hasOwnProperty(l)&&(i=n[l],i!=null)&&(l==="selected"?t.selected=i&&typeof i!="function"&&typeof i!="symbol":dt(t,e,l,i,n,null));return;case"dialog":We("beforetoggle",t),We("toggle",t),We("cancel",t),We("close",t);break;case"iframe":case"object":We("load",t);break;case"video":case"audio":for(i=0;i<zl.length;i++)We(zl[i],t);break;case"image":We("error",t),We("load",t);break;case"details":We("toggle",t);break;case"embed":case"source":case"link":We("error",t),We("load",t);case"area":case"base":case"br":case"col":case"hr":case"keygen":case"meta":case"param":case"track":case"wbr":case"menuitem":for(c in n)if(n.hasOwnProperty(c)&&(i=n[c],i!=null))switch(c){case"children":case"dangerouslySetInnerHTML":throw Error(J(137,e));default:dt(t,e,c,i,n,null)}return;default:if(Pm(e)){for(h in n)n.hasOwnProperty(h)&&(i=n[h],i!==void 0&&ym(t,e,h,i,n,void 0));return}}for(o in n)n.hasOwnProperty(o)&&(i=n[o],i!=null&&dt(t,e,o,i,n,null))}var _b={};function Sb(t,e,n,i){switch(e){case"div":case"span":case"svg":case"path":case"a":case"g":case"p":case"li":break;case"input":var s=null,r=null,a=null,o=null,l=null,c=null,h=null;for(d in n){var p=n[d];if(n.hasOwnProperty(d)&&p!=null)switch(d){case"checked":break;case"value":break;case"defaultValue":l=p;default:i.hasOwnProperty(d)||dt(t,e,d,null,i,p)}}for(var u in i){var d=i[u];if(p=n[u],i.hasOwnProperty(u)&&(d!=null||p!=null))switch(u){case"type":d!==p&&(nt=!0),r=d;break;case"name":d!==p&&(nt=!0),s=d;break;case"checked":d!==p&&(nt=!0),c=d;break;case"defaultChecked":d!==p&&(nt=!0),h=d;break;case"value":d!==p&&(nt=!0),a=d;break;case"defaultValue":d!==p&&(nt=!0),o=d;break;case"children":case"dangerouslySetInnerHTML":if(d!=null)throw Error(J(137,e));break;default:d!==p&&dt(t,e,u,d,i,p)}}Up(t,a,o,l,c,h,r,s);return;case"select":d=a=o=u=null;for(r in n)if(l=n[r],n.hasOwnProperty(r)&&l!=null)switch(r){case"value":break;case"multiple":d=l;default:i.hasOwnProperty(r)||dt(t,e,r,null,i,l)}for(s in i)if(r=i[s],l=n[s],i.hasOwnProperty(s)&&(r!=null||l!=null))switch(s){case"value":r!==l&&(nt=!0),u=r;break;case"defaultValue":r!==l&&(nt=!0),o=r;break;case"multiple":r!==l&&(nt=!0),a=r;default:r!==l&&dt(t,e,s,r,i,l)}e=o,n=a,i=d,u!=null?Ja(t,!!n,u,!1):!!i!=!!n&&(e!=null?Ja(t,!!n,e,!0):Ja(t,!!n,n?[]:"",!1));return;case"textarea":d=u=null;for(o in n)if(s=n[o],n.hasOwnProperty(o)&&s!=null&&!i.hasOwnProperty(o))switch(o){case"value":break;case"children":break;default:dt(t,e,o,null,i,s)}for(a in i)if(s=i[a],r=n[a],i.hasOwnProperty(a)&&(s!=null||r!=null))switch(a){case"value":s!==r&&(nt=!0),u=s;break;case"defaultValue":s!==r&&(nt=!0),d=s;break;case"children":break;case"dangerouslySetInnerHTML":if(s!=null)throw Error(J(91));break;default:s!==r&&dt(t,e,a,s,i,r)}n_(t,u,d);return;case"option":for(var v in n)u=n[v],n.hasOwnProperty(v)&&u!=null&&!i.hasOwnProperty(v)&&(v==="selected"?t.selected=!1:dt(t,e,v,null,i,u));for(l in i)u=i[l],d=n[l],i.hasOwnProperty(l)&&u!==d&&(u!=null||d!=null)&&(l==="selected"?(u!==d&&(nt=!0),t.selected=u&&typeof u!="function"&&typeof u!="symbol"):dt(t,e,l,u,i,d));return;case"img":case"link":case"area":case"base":case"br":case"col":case"embed":case"hr":case"keygen":case"meta":case"param":case"source":case"track":case"wbr":case"menuitem":for(var M in n)u=n[M],n.hasOwnProperty(M)&&u!=null&&!i.hasOwnProperty(M)&&dt(t,e,M,null,i,u);for(c in i)if(u=i[c],d=n[c],i.hasOwnProperty(c)&&u!==d&&(u!=null||d!=null))switch(c){case"children":case"dangerouslySetInnerHTML":if(u!=null)throw Error(J(137,e));break;default:dt(t,e,c,u,i,d)}return;default:if(Pm(e)){for(var m in n)u=n[m],n.hasOwnProperty(m)&&u!==void 0&&!i.hasOwnProperty(m)&&ym(t,e,m,void 0,i,u);for(h in i)u=i[h],d=n[h],!i.hasOwnProperty(h)||u===d||u===void 0&&d===void 0||ym(t,e,h,u,i,d);return}}for(var f in n)u=n[f],n.hasOwnProperty(f)&&u!=null&&!i.hasOwnProperty(f)&&dt(t,e,f,null,i,u);for(p in i)u=i[p],d=n[p],!i.hasOwnProperty(p)||u===d||u==null&&d==null||dt(t,e,p,u,i,d)}function uy(t){switch(t){case"css":case"script":case"font":case"img":case"image":case"input":case"link":return!0;default:return!1}}function Ab(){if(typeof performance.getEntriesByType=="function"){for(var t=0,e=0,n=performance.getEntriesByType("resource"),i=0;i<n.length;i++){var s=n[i],r=s.transferSize,a=s.initiatorType,o=s.duration;if(r&&o&&uy(a)){for(a=0,o=s.responseEnd,i+=1;i<n.length;i++){var l=n[i],c=l.startTime;if(c>o)break;var h=l.transferSize,p=l.initiatorType;h&&uy(p)&&(l=l.responseEnd,a+=h*(l<o?1:(o-c)/(l-c)))}if(--i,e+=8*(r+a)/(s.duration/1e3),t++,10<t)break}}if(0<t)return e/t/1e6}return navigator.connection&&(t=navigator.connection.downlink,typeof t=="number")?t:5}var _m=null,Sm=null;function Gl(t){return t.nodeType===9?t:t.ownerDocument}function fy(t){switch(t){case"http://www.w3.org/2000/svg":return 1;case"http://www.w3.org/1998/Math/MathML":return 2;default:return 0}}function lA(t,e){if(t===0)switch(e){case"svg":return 1;case"math":return 2;default:return 0}return t===1&&e==="foreignObject"?0:t}function cA(t,e,n,i){return n=Gl(n).createElement(t),n[mn]=i,n[qn]=e,yn(n,t,e),un(n),n}function Am(t,e){return t==="textarea"||t==="noscript"||typeof e.children=="string"||typeof e.children=="number"||typeof e.children=="bigint"||typeof e.dangerouslySetInnerHTML=="object"&&e.dangerouslySetInnerHTML!==null&&e.dangerouslySetInnerHTML.__html!=null}var gp=null;function Mb(){var t=window.event;return t&&t.type==="popstate"?t===gp?!1:(gp=t,!0):(gp=null,!1)}var Sg=typeof setTimeout=="function"?setTimeout:void 0,Eb=typeof clearTimeout=="function"?clearTimeout:void 0,hy=typeof Promise=="function"?Promise:void 0,dy=typeof requestAnimationFrame=="function"?requestAnimationFrame:Sg,Tb=typeof queueMicrotask=="function"?queueMicrotask:typeof hy<"u"?function(t){return hy.resolve(null).then(t).catch(bb)}:Sg;function bb(t){setTimeout(function(){throw t})}function Ar(t){return t==="head"}function py(t,e){var n=e,i=0;do{var s=n.nextSibling;if(t.removeChild(n),s&&s.nodeType===8)if(n=s.data,n==="/$"||n==="/&"){if(i===0){t.removeChild(s),yo(e);return}i--}else if(n==="$"||n==="$?"||n==="$~"||n==="$!"||n==="&")i++;else if(n==="html")xp(t.ownerDocument.documentElement);else if(n==="head"){n=t.ownerDocument.head,xp(n);for(var r=n.firstChild;r;){var a=r.nextSibling,o=r.nodeName;r[Kl]||o==="SCRIPT"||o==="STYLE"||o==="LINK"&&r.rel.toLowerCase()==="stylesheet"||n.removeChild(r),r=a}}else n==="body"&&xp(t.ownerDocument.body);n=s}while(n);yo(e)}function my(t,e){var n=t;t=0;do{var i=n.nextSibling;if(n.nodeType===1?e?(n._stashedDisplay=n.style.display,n.style.display="none"):(n.style.display=n._stashedDisplay||"",n.getAttribute("style")===""&&n.removeAttribute("style")):n.nodeType===3&&(e?(n._stashedText=n.nodeValue,n.nodeValue=""):n.nodeValue=n._stashedText||""),i&&i.nodeType===8)if(n=i.data,n==="/$"){if(t===0)break;t--}else n!=="$"&&n!=="$?"&&n!=="$~"&&n!=="$!"||t++;n=i}while(n)}function uA(t,e,n){if(e=CSS.escape(e)!==e?"r-"+btoa(e).replace(/=/g,""):e,t.style.viewTransitionName=e,n!=null&&(t.style.viewTransitionClass=n),n=getComputedStyle(t),n.display==="inline"){if(e=t.getClientRects(),e.length===1)var i=1;else for(var s=i=0;s<e.length;s++){var r=e[s];0<r.width&&0<r.height&&i++}i===1&&(t=t.style,t.display=e.length===1?"inline-block":"block",t.marginTop="-"+n.paddingTop,t.marginBottom="-"+n.paddingBottom)}}function fA(t,e){t=t.style,e=e.style;var n=e!=null?e.hasOwnProperty("viewTransitionName")?e.viewTransitionName:e.hasOwnProperty("view-transition-name")?e["view-transition-name"]:null:null;t.viewTransitionName=n==null||typeof n=="boolean"?"":(""+n).trim(),n=e!=null?e.hasOwnProperty("viewTransitionClass")?e.viewTransitionClass:e.hasOwnProperty("view-transition-class")?e["view-transition-class"]:null:null,t.viewTransitionClass=n==null||typeof n=="boolean"?"":(""+n).trim(),t.display==="inline-block"&&(e==null?t.display=t.margin="":(n=e.display,t.display=n==null||typeof n=="boolean"?"":n,n=e.margin,n!=null?t.margin=n:(n=e.hasOwnProperty("marginTop")?e.marginTop:e["margin-top"],t.marginTop=n==null||typeof n=="boolean"?"":n,e=e.hasOwnProperty("marginBottom")?e.marginBottom:e["margin-bottom"],t.marginBottom=e==null||typeof e=="boolean"?"":e)))}function hA(t,e,n){return n=n.ownerDocument.defaultView,{rect:t,abs:e.position==="absolute"||e.position==="fixed",clip:e.clipPath!=="none"||e.overflow!=="visible"||e.filter!=="none"||e.mask!=="none"||e.mask!=="none"||e.borderRadius!=="0px",view:0<=t.bottom&&0<=t.right&&t.top<=n.innerHeight&&t.left<=n.innerWidth}}function Mm(t){var e=t.getBoundingClientRect(),n=getComputedStyle(t);return hA(e,n,t)}function wb(t){var e=t.getBoundingClientRect();e=new DOMRect(e.x+2e4,e.y+2e4,e.width,e.height);var n=getComputedStyle(t);return hA(e,n,t)}function Cb(t){return t.documentElement.clientHeight}function Rb(t){this.addEventListener("load",t),this.addEventListener("error",t)}function Db(t,e,n,i,s,r,a,o,l){var c=e.nodeType===9?e:e.ownerDocument;try{var h=c.startViewTransition({update:function(){var u=c.defaultView,d=u.navigation&&u.navigation.transition,v=c.fonts.status;i();var M=[];if(v==="loaded"&&(Cb(c),c.fonts.status==="loading"&&M.push(c.fonts.ready)),v=M.length,t!==null)for(var m=t.suspenseyImages,f=0,g=0;g<m.length;g++){var S=m[g];if(!S.complete){var _=S.getBoundingClientRect();if(0<_.bottom&&0<_.right&&_.top<u.innerHeight&&_.left<u.innerWidth){if(f+=EA(S),f>Ru){M.length=v;break}S=new Promise(Rb.bind(S)),M.push(S)}}}if(0<M.length)return u=Promise.race([Promise.all(M),new Promise(function(E){return setTimeout(E,500)})]).then(s,s),(d?Promise.allSettled([d.finished,u]):u).then(r,r);if(s(),d)return d.finished.then(r,r);r()},types:n});c.__reactViewTransition=h;var p=[];return h.ready.then(function(){for(var u=c.documentElement.getAnimations({subtree:!0}),d=0;d<u.length;d++){var v=u[d],M=v.effect,m=M.pseudoElement;if(m!=null&&m.startsWith("::view-transition")){p.push(v),v=M.getKeyframes();for(var f=m=void 0,g=!0,S=0;S<v.length;S++){var _=v[S],E=_.width;if(m===void 0)m=E;else if(m!==E){g=!1;break}if(E=_.height,f===void 0)f=E;else if(f!==E){g=!1;break}delete _.width,delete _.height,_.transform==="none"&&delete _.transform}g&&m!==void 0&&f!==void 0&&(M.setKeyframes(v),g=getComputedStyle(M.target,M.pseudoElement),g.width!==m||g.height!==f)&&(g=v[0],g.width=m,g.height=f,g=v[v.length-1],g.width=m,g.height=f,M.setKeyframes(v))}}a()},function(u){c.__reactViewTransition===h&&(c.__reactViewTransition=null);try{typeof u=="object"&&u!==null&&u.name==="InvalidStateError"&&(u.message==="View transition was skipped because document visibility state is hidden."||u.message==="Skipping view transition because document visibility state has become hidden."||u.message==="Skipping view transition because viewport size changed."||u.message==="Transition was aborted because of invalid state")&&(u=null),u!==null&&l(u)}finally{i(),s(),a()}}),h.finished.finally(function(){for(var u=0;u<p.length;u++)p[u].cancel();c.__reactViewTransition===h&&(c.__reactViewTransition=null),o()}),h}catch{return i(),s(),a(),null}}function Qr(t,e){this._scope=document.documentElement,this._selector="::view-transition-"+t+"("+e+")"}Qr.prototype.animate=function(t,e){return e=typeof e=="number"?{duration:e}:yt({},e),e.pseudoElement=this._selector,this._scope.animate(t,e)};Qr.prototype.getAnimations=function(){for(var t=this._scope,e=this._selector,n=t.getAnimations({subtree:!0}),i=[],s=0;s<n.length;s++){var r=n[s].effect;r!==null&&r.target===t&&r.pseudoElement===e&&i.push(n[s])}return i};Qr.prototype.getComputedStyle=function(){return getComputedStyle(this._scope,this._selector)};function dA(t){return{name:t,group:new Qr("group",t),imagePair:new Qr("image-pair",t),old:new Qr("old",t),new:new Qr("new",t)}}function oi(t){this._fragmentFiber=t,this._observers=this._eventListeners=null}oi.prototype.addEventListener=function(t,e,n){var i=null,s=null;if(!(n!=null&&typeof n!="boolean"&&(i=n.signal||null,i!==null&&i.aborted))){this._eventListeners===null&&(this._eventListeners=[]);var r=this._eventListeners;if(pA(r,t,e,n)===-1){var a=this,o=e;n!=null&&typeof n!="boolean"&&n.once===!0&&(o=function(l){a.removeEventListener(t,e,n),typeof e=="function"?e.call(this,l):e.handleEvent(l)}),i!==null&&(s=a.removeEventListener.bind(a,t,e,n),i.addEventListener("abort",s,{once:!0}),s=i.removeEventListener.bind(i,"abort",s)),i=mo(n),r.push({type:t,listener:e,optionsOrUseCapture:n,attachedListener:o,cleanup:s}),Yn(this._fragmentFiber.child,!1,Ub,t,o,i)}this._eventListeners=r}};function Ub(t,e,n,i){return nn(t).addEventListener(e,n,i),!1}oi.prototype.removeEventListener=function(t,e,n){var i=this._eventListeners;if(i!==null&&(e=pA(i,t,e,n),e!==-1)){var s=i[e];n=s.attachedListener;var r=s.cleanup;s=mo(s.optionsOrUseCapture),Yn(this._fragmentFiber.child,!1,Ib,t,n,s),i.splice(e,1),r!==null&&r()}};function Ib(t,e,n,i){return nn(t).removeEventListener(e,n,i),!1}function mo(t){return t!=null&&typeof t!="boolean"&&(t.once===!0||t.signal instanceof AbortSignal)?{capture:t.capture,passive:t.passive}:t}function gy(t){return t==null?"c=0":typeof t=="boolean"?"c="+(t?"1":"0"):"c="+(t.capture?"1":"0")}function pA(t,e,n,i){if(t.length===0)return-1;i=gy(i);for(var s=0;s<t.length;s++){var r=t[s];if(r.type===e&&r.listener===n&&gy(r.optionsOrUseCapture)===i)return s}return-1}oi.prototype.dispatchEvent=function(t){var e=la(this._fragmentFiber);if(e===null)return!0;e=nn(e);var n=this._eventListeners;if(n!==null&&0<n.length||!t.bubbles){var i=e.nodeType===9?e.createComment(""):document.createTextNode("");if(n)for(var s=0;s<n.length;s++){var r=n[s];i.addEventListener(r.type,r.attachedListener,mo(r.optionsOrUseCapture))}if(e.appendChild(i),t=i.dispatchEvent(t),n)for(s=0;s<n.length;s++)r=n[s],i.removeEventListener(r.type,r.attachedListener,mo(r.optionsOrUseCapture));return e.removeChild(i),t}return e.dispatchEvent(t)};oi.prototype.focus=function(t){Yn(this._fragmentFiber.child,!0,mA,t,void 0,void 0)};function mA(t,e){return t.tag===6?!1:(t=nn(t),Wb(t,e))}oi.prototype.focusLast=function(t){var e=[];Yn(this._fragmentFiber.child,!0,Ag,e,void 0,void 0);for(var n=e.length-1;0<=n&&!mA(e[n],t);n--);};function Ag(t,e){return e.push(t),!1}oi.prototype.blur=function(){var t=la(this._fragmentFiber);t!==null&&(t=nn(t),t=Gl(t).activeElement,t!==null&&Yn(this._fragmentFiber.child,!1,Bb,t,void 0,void 0))};function Bb(t,e){return t.tag===6?!1:(t=nn(t),t===e||t.contains(e)?(e.blur(),!0):!1)}oi.prototype.observeUsing=function(t){this._observers===null&&(this._observers=new Set),this._observers.add(t),Yn(this._fragmentFiber.child,!1,Nb,t,void 0,void 0)};function Nb(t,e){return t.tag===6||(t=nn(t),e.observe(t)),!1}oi.prototype.unobserveUsing=function(t){var e=this._observers;if(e!==null&&e.has(t)){e.delete(t),Yn(this._fragmentFiber.child,!1,Pb,t,void 0,void 0);for(var n=e=0;n<Ni.length;n++){var i=Ni[n];i.fragmentInstance===this&&i.observer===t?t.unobserve(i.instance):Ni[e++]=i}Ni.length=e}};function Pb(t,e){return t.tag===6||(t=nn(t),e.unobserve(t)),!1}var Ni=[],vp=!1;function Lb(t,e,n){Ni.push({fragmentInstance:t,observer:e,instance:n}),vp||(vp=!0,Xb(function(){vp=!1;var i=Ni;Ni=[];for(var s=0;s<i.length;s++){var r=i[s];r.observer.unobserve(r.instance)}}))}oi.prototype.getClientRects=function(){var t=[];return Yn(this._fragmentFiber.child,!1,Ob,t,void 0,void 0),t};function Ob(t,e){if(t.tag===6){t=t.stateNode;var n=t.ownerDocument.createRange();n.selectNodeContents(t),e.push.apply(e,n.getClientRects())}else t=nn(t),e.push.apply(e,t.getClientRects());return!1}oi.prototype.getRootNode=function(t){var e=la(this._fragmentFiber);return e===null?this:nn(e).getRootNode(t)};oi.prototype.compareDocumentPosition=function(t){var e=la(this._fragmentFiber);if(e===null)return Node.DOCUMENT_POSITION_DISCONNECTED;var n=[];Yn(this._fragmentFiber.child,!1,Ag,n,void 0,void 0);var i=nn(e);if(n.length===0){if(n=i,Zv(this._fragmentFiber)){e:{for(e=this._fragmentFiber.return;e!==null;){if(e.tag===4){e=e.stateNode.containerInfo;break e}if(e.tag===3||e.tag===5||e.tag===27)break;e=e.return}e=null}e!=null&&(n=e)}e=this._fragmentFiber;var s=i=n.compareDocumentPosition(t);return n===t?s=Node.DOCUMENT_POSITION_CONTAINS:i&Node.DOCUMENT_POSITION_CONTAINED_BY&&(n=Oy(e)[1],n===null?s=Node.DOCUMENT_POSITION_PRECEDING:(t=nn(n).compareDocumentPosition(t),s=t===0||t&Node.DOCUMENT_POSITION_FOLLOWING?Node.DOCUMENT_POSITION_FOLLOWING:Node.DOCUMENT_POSITION_PRECEDING)),s|=Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC}e=nn(n[0]),s=nn(n[n.length-1]);var r=Zv(this._fragmentFiber)?e.parentElement:i;if(r==null)return Node.DOCUMENT_POSITION_DISCONNECTED;i=r.compareDocumentPosition(e)&Node.DOCUMENT_POSITION_CONTAINED_BY,r=r.compareDocumentPosition(s)&Node.DOCUMENT_POSITION_CONTAINED_BY;var a=e.compareDocumentPosition(t),o=s.compareDocumentPosition(t),l=a&Node.DOCUMENT_POSITION_CONTAINED_BY||o&Node.DOCUMENT_POSITION_CONTAINED_BY;return o=i&&r&&a&Node.DOCUMENT_POSITION_FOLLOWING&&o&Node.DOCUMENT_POSITION_PRECEDING,e=i&&e===t||r&&s===t||l||o?Node.DOCUMENT_POSITION_CONTAINED_BY:!i&&e===t||!r&&s===t?Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC:a,e&Node.DOCUMENT_POSITION_DISCONNECTED||e&Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC||Fb(e,this._fragmentFiber,n[0],n[n.length-1],t)?e:Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC};function Fb(t,e,n,i,s){var r=qr(s);if(t&Node.DOCUMENT_POSITION_CONTAINED_BY){if(n=!!r)e:{for(;r!==null;){if(r.tag===7&&(r===e||r.alternate===e)){n=!0;break e}r=r.return}n=!1}return n}if(t&Node.DOCUMENT_POSITION_CONTAINS){if(r===null)return r=s.ownerDocument,s===r||s===r.documentElement||s===r.body;e:{for(r=e,e=la(e);r!==null;){if(!(r.tag!==5&&r.tag!==3&&r.tag!==27||r!==e&&r.alternate!==e)){r=!0;break e}r=r.return}r=!1}return r}return t&Node.DOCUMENT_POSITION_PRECEDING?((e=!!r)&&!(e=r===n)&&(e=_p(n,r,Kv),e===null?e=!1:(Yn(e,!0,pE,r,n),r=Fa,Fa=null,e=r!==null)),e):t&Node.DOCUMENT_POSITION_FOLLOWING?((e=!!r)&&!(e=r===i)&&(e=_p(i,r,Kv),e===null?e=!1:(Yn(e,!0,mE,r,i),r=Fa,yp=Fa=null,e=r!==null)),e):!1}function vy(t,e){var n=t.ownerDocument.createRange();n.selectNodeContents(t),t=n.getBoundingClientRect(),window.scrollTo(window.scrollX+t.left,e?window.scrollY+t.top:window.scrollY+t.bottom-window.innerHeight)}oi.prototype.scrollIntoView=function(t){if(typeof t=="object")throw Error(J(566));var e=[];Yn(this._fragmentFiber.child,!1,Ag,e,void 0,void 0);var n=t!==!1;if(e.length===0){var i=Oy(this._fragmentFiber);if(i=n?i[1]||i[0]||la(this._fragmentFiber):i[0]||i[1],i===null)return;if(i.tag===6){t=nn(i),vy(t,n);return}if(i=nn(i),i.nodeType!==9){if(i.nodeType===11){n="host"in i?i.host:null,n!==null&&n.scrollIntoView(t);return}i.scrollIntoView(t)}}for(i=n?e.length-1:0;i!==(n?-1:e.length);){var s=e[i];s.tag===6?(s=nn(s),vy(s,n)):nn(s).scrollIntoView(t),i+=n?-1:1}};function zb(t,e){return t=nn(t),gA(t,e),!1}function gA(t,e){t.reactFragments==null&&(t.reactFragments=new Set),t.reactFragments.add(e)}function vA(t,e){var n=e._eventListeners;if(n!==null)for(var i=0;i<n.length;i++){var s=n[i];t.addEventListener(s.type,s.attachedListener,mo(s.optionsOrUseCapture))}t.nodeType!==3&&(n=e._observers,n!==null&&n.forEach(function(r){for(var a=0,o=0;o<Ni.length;o++){var l=Ni[o];(l.fragmentInstance!==e||l.observer!==r||l.instance!==t)&&(Ni[a++]=l)}Ni.length=a,r.observe(t)}),gA(t,e))}function Hb(t,e){var n=e._eventListeners;if(n!==null)for(var i=0;i<n.length;i++){var s=n[i];t.removeEventListener(s.type,s.attachedListener,mo(s.optionsOrUseCapture))}t.nodeType!==3&&(n=e._observers,n!==null&&n.forEach(function(r){typeof r.rootMargin=="string"?Lb(e,r,t):r.unobserve(t)}),t.reactFragments!=null&&t.reactFragments.delete(e))}function Em(t){var e=t.firstChild;for(e&&e.nodeType===10&&(e=e.nextSibling);e;){var n=e;switch(e=e.nextSibling,n.nodeName){case"HTML":case"HEAD":case"BODY":Em(n),cf(n);continue;case"SCRIPT":case"STYLE":continue;case"LINK":if(n.rel.toLowerCase()==="stylesheet")continue}t.removeChild(n)}}function Gb(t,e,n,i){for(;t.nodeType===1;){var s=n;if(t.nodeName.toLowerCase()!==e.toLowerCase()){if(!i&&(t.nodeName!=="INPUT"||t.type!=="hidden"))break}else if(i){if(!t[Kl])switch(e){case"meta":if(!t.hasAttribute("itemprop"))break;return t;case"link":if(r=t.getAttribute("rel"),r==="stylesheet"&&t.hasAttribute("data-precedence"))break;if(r!==s.rel||t.getAttribute("href")!==(s.href==null||s.href===""?null:s.href)||t.getAttribute("crossorigin")!==(s.crossOrigin==null?null:s.crossOrigin)||t.getAttribute("title")!==(s.title==null?null:s.title))break;return t;case"style":if(t.hasAttribute("data-precedence"))break;return t;case"script":if(r=t.getAttribute("src"),(r!==(s.src==null?null:s.src)||t.getAttribute("type")!==(s.type==null?null:s.type)||t.getAttribute("crossorigin")!==(s.crossOrigin==null?null:s.crossOrigin))&&r&&t.hasAttribute("async")&&!t.hasAttribute("itemprop"))break;return t;default:return t}}else if(e==="input"&&t.type==="hidden"){var r=s.name==null?null:""+s.name;if(s.type==="hidden"&&t.getAttribute("name")===r)return t}else return t;if(t=Ai(t.nextSibling),t===null)break}return null}function Vb(t,e,n){if(e==="")return null;for(;t.nodeType!==3;)if((t.nodeType!==1||t.nodeName!=="INPUT"||t.type!=="hidden")&&!n||(t=Ai(t.nextSibling),t===null))return null;return t}function xA(t,e){for(;t.nodeType!==8;)if((t.nodeType!==1||t.nodeName!=="INPUT"||t.type!=="hidden")&&!e||(t=Ai(t.nextSibling),t===null))return null;return t}function Tm(t){return t.data==="$?"||t.data==="$~"}function Mg(t){return t.data==="$!"||t.data==="$?"&&t.ownerDocument.readyState!=="loading"}function kb(t,e){var n=t.ownerDocument;if(t.data==="$~")t._reactRetry=e;else if(t.data!=="$?"||n.readyState!=="loading")e();else{var i=function(){e(),n.removeEventListener("DOMContentLoaded",i)};n.addEventListener("DOMContentLoaded",i),t._reactRetry=i}}function Ai(t){for(;t!=null;t=t.nextSibling){var e=t.nodeType;if(e===1||e===3)break;if(e===8){if(e=t.data,e==="$"||e==="$!"||e==="$?"||e==="$~"||e==="&"||e==="F!"||e==="F")break;if(e==="/$"||e==="/&")return null}}return t}var bm=null;function xy(t){t=t.nextSibling;for(var e=0;t;){if(t.nodeType===8){var n=t.data;if(n==="/$"||n==="/&"){if(e===0)return Ai(t.nextSibling);e--}else n!=="$"&&n!=="$!"&&n!=="$?"&&n!=="$~"&&n!=="&"||e++}t=t.nextSibling}return null}function yy(t){t=t.previousSibling;for(var e=0;t;){if(t.nodeType===8){var n=t.data;if(n==="$"||n==="$!"||n==="$?"||n==="$~"||n==="&"){if(e===0)return t;e--}else n!=="/$"&&n!=="/&"||e++}t=t.previousSibling}return null}function Wb(t,e){function n(){i=!0}if(t.ownerDocument.activeElement===t)return!0;var i=!1;try{t.ownerDocument.addEventListener("focus",n,!0),(t.focus||HTMLElement.prototype.focus).call(t,e)}finally{t.ownerDocument.removeEventListener("focus",n,!0)}return i}function Xb(t){dy(function(){dy(function(e){return t(e)})})}function yA(t,e,n){switch(e=Gl(n),t){case"html":if(t=e.documentElement,!t)throw Error(J(452));return t;case"head":if(t=e.head,!t)throw Error(J(453));return t;case"body":if(t=e.body,!t)throw Error(J(454));return t;default:throw Error(J(451))}}function _A(t,e,n){for(var i in n){var s=n[i];n.hasOwnProperty(i)&&s!=null&&dt(t,e,i,null,_b,s)}n.dangerouslySetInnerHTML!=null&&(t.textContent=""),t.onclick===ns&&(t.onclick=null),cf(t)}function xp(t){for(var e=t.attributes;e.length;)t.removeAttributeNode(e[0]);cf(t)}var Mi=new Map,_y=new Set;function Vl(t){if(typeof t.getRootNode=="function"){var e=t.getRootNode();if(e.nodeType===9||e.nodeType===11)return e}return t.nodeType===9?t:t.ownerDocument}var Ns=rt.d;rt.d={f:Yb,r:qb,D:Qb,C:Zb,L:Kb,m:Jb,X:$b,S:jb,M:ew};function Yb(){var t=Ns.f(),e=Mf();return t||e}function qb(t){var e=So(t);e!==null&&e.tag===5&&e.type==="form"?sS(e):Ns.r(t)}var To=typeof document>"u"?null:document;function SA(t,e,n){var i=To;if(i&&typeof e=="string"&&e){var s=yi(e);s='link[rel="'+t+'"][href="'+s+'"]',typeof n=="string"&&(s+='[crossorigin="'+n+'"]'),_y.has(s)||(_y.add(s),t={rel:t,crossOrigin:n,href:e},i.querySelector(s)===null&&(e=i.createElement("link"),yn(e,"link",t),un(e),i.head.appendChild(e)))}}function Qb(t){Ns.D(t),SA("dns-prefetch",t,null)}function Zb(t,e){Ns.C(t,e),SA("preconnect",t,e)}function Kb(t,e,n){Ns.L(t,e,n);var i=To;if(i&&t&&e){var s='link[rel="preload"][as="'+yi(e)+'"]';e==="image"&&n&&n.imageSrcSet?(s+='[imagesrcset="'+yi(n.imageSrcSet)+'"]',typeof n.imageSizes=="string"&&(s+='[imagesizes="'+yi(n.imageSizes)+'"]')):s+='[href="'+yi(t)+'"]';var r=s;switch(e){case"style":r=go(t);break;case"script":r=bo(t)}if(!(Mi.has(r)||(t=yt({rel:"preload",href:e==="image"&&n&&n.imageSrcSet?void 0:t,as:e},n),Mi.set(r,t),i.querySelector(s)!==null||e==="style"&&i.querySelector(nc(r))||e==="script"&&i.querySelector(ic(r))))){var a=i.createElement("link");yn(a,"link",t),e==="style"&&(a[Lu]=!0,a.onload=a.onerror=function(){Ky(a)}),un(a),i.head.appendChild(a)}}}function Jb(t,e){Ns.m(t,e);var n=To;if(n&&t){var i=e&&typeof e.as=="string"?e.as:"script",s='link[rel="modulepreload"][as="'+yi(i)+'"][href="'+yi(t)+'"]',r=s;switch(i){case"audioworklet":case"paintworklet":case"serviceworker":case"sharedworker":case"worker":case"script":r=bo(t)}if(!Mi.has(r)&&(t=yt({rel:"modulepreload",href:t},e),Mi.set(r,t),n.querySelector(s)===null)){switch(i){case"audioworklet":case"paintworklet":case"serviceworker":case"sharedworker":case"worker":case"script":if(n.querySelector(ic(r)))return}i=n.createElement("link"),yn(i,"link",t),un(i),n.head.appendChild(i)}}}function jb(t,e,n){Ns.S(t,e,n);var i=To;if(i&&t){var s=Ka(i).hoistableStyles,r=go(t);e=e||"default";var a=s.get(r);if(!a){var o={loading:0,preload:null};if(a=i.querySelector(nc(r)))o.loading=5;else{t=yt({rel:"stylesheet",href:t,"data-precedence":e},n),(n=Mi.get(r))&&Eg(t,n);var l=a=i.createElement("link");un(l),yn(l,"link",t),l._p=new Promise(function(c,h){l.onload=c,l.onerror=h}),l.addEventListener("load",function(){o.loading|=1}),l.addEventListener("error",function(){o.loading|=2}),o.loading|=4,wu(a,e,i)}a={type:"stylesheet",instance:a,count:1,state:o},s.set(r,a)}}}function $b(t,e){Ns.X(t,e);var n=To;if(n&&t){var i=Ka(n).hoistableScripts,s=bo(t),r=i.get(s);r||(r=n.querySelector(ic(s)),r||(t=yt({src:t,async:!0},e),(e=Mi.get(s))&&Tg(t,e),r=n.createElement("script"),un(r),yn(r,"link",t),n.head.appendChild(r)),r={type:"script",instance:r,count:1,state:null},i.set(s,r))}}function ew(t,e){Ns.M(t,e);var n=To;if(n&&t){var i=Ka(n).hoistableScripts,s=bo(t),r=i.get(s);r||(r=n.querySelector(ic(s)),r||(t=yt({src:t,async:!0,type:"module"},e),(e=Mi.get(s))&&Tg(t,e),r=n.createElement("script"),un(r),yn(r,"link",t),n.head.appendChild(r)),r={type:"script",instance:r,count:1,state:null},i.set(s,r))}}function Sy(t,e,n,i){var s=(s=ar.current)?Vl(s):null;if(!s)throw Error(J(446));switch(t){case"meta":case"title":return null;case"style":return typeof n.precedence=="string"&&typeof n.href=="string"?(n=go(n.href),e=Ka(s).hoistableStyles,i=e.get(n),i||(i={type:"style",instance:null,count:0,state:null},e.set(n,i)),i):{type:"void",instance:null,count:0,state:null};case"link":if(n.rel==="stylesheet"&&typeof n.href=="string"&&typeof n.precedence=="string"){t=go(n.href);var r=Ka(s).hoistableStyles,a=r.get(t);if(a||(s=s.ownerDocument||s,a={type:"stylesheet",instance:null,count:0,state:{loading:0,preload:null}},r.set(t,a),(r=s.querySelector(nc(t)))?r._p||(a.instance=r,a.state.loading=5):(r=Mi.get(t),r||(r={rel:"preload",as:"style",href:n.href,crossOrigin:n.crossOrigin,integrity:n.integrity,media:n.media,hrefLang:n.hrefLang,referrerPolicy:n.referrerPolicy},Mi.set(t,r)),tw(s,t,r,a.state))),e&&i===null)throw Error(J(528,""));return a}if(e&&i!==null)throw Error(J(529,""));return null;case"script":return e=n.async,n=n.src,typeof n=="string"&&e&&typeof e!="function"&&typeof e!="symbol"?(n=bo(n),e=Ka(s).hoistableScripts,i=e.get(n),i||(i={type:"script",instance:null,count:0,state:null},e.set(n,i)),i):{type:"void",instance:null,count:0,state:null};default:throw Error(J(444,t))}}function go(t){return'href="'+yi(t)+'"'}function nc(t){return'link[rel="stylesheet"]['+t+"]"}function AA(t){return yt({},t,{"data-precedence":t.precedence,precedence:null})}function tw(t,e,n,i){if(e=t.querySelector('link[rel="preload"][as="style"]['+e+"]")){if(e[Lu]!==!0){i.loading=1;return}}else e=t.createElement("link"),e[Lu]=!0,e.onload=e.onerror=Ky.bind(null,e),yn(e,"link",n),un(e),t.head.appendChild(e);i.preload=e,e.addEventListener("load",function(){return i.loading|=1}),e.addEventListener("error",function(){return i.loading|=2})}function bo(t){return'[src="'+yi(t)+'"]'}function ic(t){return"script[async]"+t}function Ay(t,e,n){if(e.count++,e.instance===null)switch(e.type){case"style":var i=t.querySelector('style[data-href~="'+yi(n.href)+'"]');if(i)return e.instance=i,un(i),i;var s=yt({},n,{"data-href":n.href,"data-precedence":n.precedence,href:null,precedence:null});return i=(t.ownerDocument||t).createElement("style"),un(i),yn(i,"style",s),wu(i,n.precedence,t),e.instance=i;case"stylesheet":s=go(n.href);var r=t.querySelector(nc(s));if(r)return e.state.loading|=4,e.instance=r,un(r),r;i=AA(n),(s=Mi.get(s))&&Eg(i,s),r=(t.ownerDocument||t).createElement("link"),un(r);var a=r;return a._p=new Promise(function(o,l){a.onload=o,a.onerror=l}),yn(r,"link",i),e.state.loading|=4,wu(r,n.precedence,t),e.instance=r;case"script":return r=bo(n.src),(s=t.querySelector(ic(r)))?(e.instance=s,un(s),s):(i=n,(s=Mi.get(r))&&(i=yt({},n),Tg(i,s)),t=t.ownerDocument||t,s=t.createElement("script"),un(s),yn(s,"link",i),t.head.appendChild(s),e.instance=s);case"void":return null;default:throw Error(J(443,e.type))}else e.type==="stylesheet"&&(e.state.loading&4)===0&&(i=e.instance,e.state.loading|=4,wu(i,n.precedence,t));return e.instance}function wu(t,e,n){for(var i=n.querySelectorAll('link[rel="stylesheet"][data-precedence],style[data-precedence]'),s=i.length?i[i.length-1]:null,r=s,a=0;a<i.length;a++){var o=i[a];if(o.dataset.precedence===e)r=o;else if(r!==s)break}r?r.parentNode.insertBefore(t,r.nextSibling):(e=n.nodeType===9?n.head:n,e.insertBefore(t,e.firstChild))}function Eg(t,e){t.crossOrigin==null&&(t.crossOrigin=e.crossOrigin),t.referrerPolicy==null&&(t.referrerPolicy=e.referrerPolicy),t.title==null&&(t.title=e.title)}function Tg(t,e){t.crossOrigin==null&&(t.crossOrigin=e.crossOrigin),t.referrerPolicy==null&&(t.referrerPolicy=e.referrerPolicy),t.integrity==null&&(t.integrity=e.integrity)}var Cu=null;function My(t,e,n){if(Cu===null){var i=new Map,s=Cu=new Map;s.set(n,i)}else s=Cu,i=s.get(n),i||(i=new Map,s.set(n,i));if(i.has(t))return i;for(i.set(t,null),n=n.getElementsByTagName(t),s=0;s<n.length;s++){var r=n[s];if(!(r[Kl]||r[mn]||t==="link"&&r.getAttribute("rel")==="stylesheet")&&r.namespaceURI!=="http://www.w3.org/2000/svg"){var a=r.getAttribute(e)||"";a=t+a;var o=i.get(a);o?o.push(r):i.set(a,[r])}}return i}function wm(t,e,n){t=t.ownerDocument||t,t.head.insertBefore(n,e==="title"?t.querySelector("head > title"):null)}function nw(t,e,n){if(n===1||e.itemProp!=null)return!1;switch(t){case"meta":case"title":return!0;case"style":if(typeof e.precedence!="string"||typeof e.href!="string"||e.href==="")break;return!0;case"link":if(typeof e.rel!="string"||typeof e.href!="string"||e.href===""||e.onLoad||e.onError)break;return e.rel==="stylesheet"?(t=e.disabled,typeof e.precedence=="string"&&t==null):!0;case"script":if(e.async&&typeof e.async!="function"&&typeof e.async!="symbol"&&!e.onLoad&&!e.onError&&e.src&&typeof e.src=="string")return!0}return!1}function Ey(t,e){return t==="img"&&e.src!=null&&e.src!==""&&e.onLoad==null&&e.loading!=="lazy"}function MA(t){return!(t.type==="stylesheet"&&(t.state.loading&3)===0)}function EA(t){return(t.width||100)*(t.height||100)*(typeof devicePixelRatio=="number"?devicePixelRatio:1)*.25}function Ty(t,e){typeof e.decode=="function"&&(t.imgCount++,e.complete||(t.imgBytes+=EA(e),t.suspenseyImages.push(e)),t=rw.bind(t),e.decode().then(t,t))}function iw(t,e,n,i){if(n.type==="stylesheet"&&(typeof i.media!="string"||matchMedia(i.media).matches!==!1)&&(n.state.loading&4)===0){if(n.instance===null){var s=go(i.href),r=e.querySelector(nc(s));if(r){e=r._p,e!==null&&typeof e=="object"&&typeof e.then=="function"&&(t.count++,t=kl.bind(t),e.then(t,t)),n.state.loading|=4,n.instance=r,un(r);return}r=e.ownerDocument||e,i=AA(i),(s=Mi.get(s))&&Eg(i,s),r=r.createElement("link"),un(r);var a=r;a._p=new Promise(function(o,l){a.onload=o,a.onerror=l}),yn(r,"link",i),n.instance=r}t.stylesheets===null&&(t.stylesheets=new Map),t.stylesheets.set(n,e),(e=n.state.preload)&&(n.state.loading&3)===0&&(t.count++,n=kl.bind(t),e.addEventListener("load",n),e.addEventListener("error",n))}}var Ru=0;function sw(t,e){return t.stylesheets&&t.count===0&&Du(t,t.stylesheets),0<t.count||0<t.imgCount?function(n){var i=setTimeout(function(){if(t.stylesheets&&Du(t,t.stylesheets),t.unsuspend){var r=t.unsuspend;t.unsuspend=null,r()}},6e4+e);0<t.imgBytes&&Ru===0&&(Ru=62500*Ab());var s=setTimeout(function(){if(t.waitingForImages=!1,t.count===0&&(t.stylesheets&&Du(t,t.stylesheets),t.unsuspend)){var r=t.unsuspend;t.unsuspend=null,r()}},(t.imgBytes>Ru?50:800)+e);return t.unsuspend=n,function(){t.unsuspend=null,clearTimeout(i),clearTimeout(s)}}:null}function TA(t){if(t.count===0&&(t.imgCount===0||!t.waitingForImages)){if(t.stylesheets)Du(t,t.stylesheets);else if(t.unsuspend){var e=t.unsuspend;t.unsuspend=null,e()}}}function kl(){this.count--,TA(this)}function rw(){this.imgCount--,TA(this)}var af=null;function Du(t,e){t.stylesheets=null,t.unsuspend!==null&&(t.count++,af=new Map,e.forEach(aw,t),af=null,kl.call(t))}function aw(t,e){if(!(e.state.loading&4)){var n=af.get(t);if(n)var i=n.get(null);else{n=new Map,af.set(t,n);for(var s=t.querySelectorAll("link[data-precedence],style[data-precedence]"),r=0;r<s.length;r++){var a=s[r];(a.nodeName==="LINK"||a.getAttribute("media")!=="not all")&&(n.set(a.dataset.precedence,a),i=a)}i&&n.set(null,i)}s=e.instance,a=s.getAttribute("data-precedence"),r=n.get(a)||i,r===i&&n.set(null,s),n.set(a,s),this.count++,i=kl.bind(this),s.addEventListener("load",i),s.addEventListener("error",i),r?r.parentNode.insertBefore(s,r.nextSibling):(t=t.nodeType===9?t.head:t,t.insertBefore(s,t.firstChild)),e.state.loading|=4}}var vo={$$typeof:ts,Provider:null,Consumer:null,_currentValue:Zr,_currentValue2:Zr,_threadCount:0};function ow(t,e,n,i,s,r,a,o,l){this.tag=1,this.containerInfo=t,this.pingCache=this.current=this.pendingChildren=null,this.timeoutHandle=-1,this.callbackNode=this.next=this.pendingContext=this.context=this.cancelPendingCommit=null,this.callbackPriority=0,this.expirationTimes=Yd(-1),this.entangledLanes=this.shellSuspendCounter=this.errorRecoveryDisabledLanes=this.expiredLanes=this.warmLanes=this.pingedLanes=this.suspendedLanes=this.pendingLanes=0,this.entanglements=Yd(0),this.hiddenUpdates=Yd(null),this.identifierPrefix=i,this.onUncaughtError=s,this.onCaughtError=r,this.onRecoverableError=a,this.pooledCache=null,this.pooledCacheLanes=0,this.formState=l,this.transitionTypes=null,this.incompleteTransitions=new Map}function bA(t,e,n,i,s,r,a,o,l,c,h,p){return t=new ow(t,e,n,a,l,c,h,p,o),e=1,r===!0&&(e|=24),r=Wn(3,null,null,e),t.current=r,r.stateNode=t,e=Xm(),e.refCount++,t.pooledCache=e,e.refCount++,r.memoizedState={element:i,isDehydrated:n,cache:e},Qm(r),t}function wA(t){return t?(t=Ya,t):Ya}function CA(t,e,n,i,s,r){s=wA(s),i.context===null?i.context=s:i.pendingContext=s,i=lr(e),i.payload={element:n},r=r===void 0?null:r,r!==null&&(i.callback=r),n=cr(t,i,e),n!==null&&(Xn(n,t,e),Sl(n,t,e))}function by(t,e){if(t=t.memoizedState,t!==null&&t.dehydrated!==null){var n=t.retryLane;t.retryLane=n!==0&&n<e?n:e}}function bg(t,e){by(t,e),(t=t.alternate)&&by(t,e)}function RA(t){if(t.tag===13||t.tag===31){var e=fa(t,67108864);e!==null&&Xn(e,t,67108864),bg(t,67108864)}}function wy(t){if(t.tag===13||t.tag===31){var e=ri();e=Bm(e);var n=fa(t,e);n!==null&&Xn(n,t,e),bg(t,e)}}var xo=!0;function lw(t,e,n,i){var s=Ue.T;Ue.T=null;var r=rt.p;try{rt.p=2,wg(t,e,n,i)}finally{rt.p=r,Ue.T=s}}function cw(t,e,n,i){var s=Ue.T;Ue.T=null;var r=rt.p;try{rt.p=8,wg(t,e,n,i)}finally{rt.p=r,Ue.T=s}}function wg(t,e,n,i){if(xo){var s=Cm(i);if(s===null)mp(t,e,i,of,n),Cy(t,i);else if(fw(s,t,e,n,i))i.stopPropagation();else if(Cy(t,i),e&4&&-1<uw.indexOf(t)){for(;s!==null;){var r=So(s);if(r!==null)switch(r.tag){case 3:if(r=r.stateNode,r.current.memoizedState.isDehydrated){var a=Wr(r.pendingLanes);if(a!==0){var o=r;for(o.pendingLanes|=2,o.entangledLanes|=2;a;){var l=1<<31-si(a);o.entanglements[1]|=l,a&=~l}cs(r),(st&6)===0&&(ef=ni()+500,tc(0,!1))}}break;case 31:case 13:o=fa(r,2),o!==null&&Xn(o,r,2),Mf(),bg(r,2)}if(r=Cm(i),r===null&&mp(t,e,i,of,n),r===s)break;s=r}s!==null&&i.stopPropagation()}else mp(t,e,i,null,n)}}function Cm(t){return t=Lm(t),Cg(t)}var of=null;function Cg(t){if(of=null,t=qr(t),t!==null){var e=Yl(t);if(e===null)t=null;else{var n=e.tag;if(n===13){if(t=Ny(e),t!==null)return t;t=null}else if(n===31){if(t=Py(e),t!==null)return t;t=null}else if(n===3){if(e.stateNode.current.memoizedState.isDehydrated)return e.tag===3?e.stateNode.containerInfo:null;t=null}else e!==t&&(t=null)}}return of=t,null}function DA(t){switch(t){case"beforetoggle":case"cancel":case"click":case"close":case"contextmenu":case"copy":case"cut":case"auxclick":case"dblclick":case"dragend":case"dragstart":case"drop":case"focusin":case"focusout":case"input":case"invalid":case"keydown":case"keypress":case"keyup":case"mousedown":case"mouseup":case"paste":case"pause":case"play":case"pointercancel":case"pointerdown":case"pointerup":case"ratechange":case"reset":case"seeked":case"submit":case"toggle":case"touchcancel":case"touchend":case"touchstart":case"volumechange":case"change":case"selectionchange":case"textInput":case"compositionstart":case"compositionend":case"compositionupdate":case"beforeblur":case"afterblur":case"beforeinput":case"blur":case"fullscreenchange":case"fullscreenerror":case"focus":case"hashchange":case"popstate":case"select":case"selectstart":return 2;case"drag":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"mousemove":case"mouseout":case"mouseover":case"pointermove":case"pointerout":case"pointerover":case"resize":case"scroll":case"touchmove":case"wheel":case"mouseenter":case"mouseleave":case"pointerenter":case"pointerleave":return 8;case"message":switch(EE()){case Gy:return 2;case Vy:return 8;case Pu:case TE:return 32;case ky:return 268435456;default:return 32}default:return 32}}var Rm=!1,dr=null,pr=null,mr=null,Wl=new Map,Xl=new Map,$s=[],uw="mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset".split(" ");function Cy(t,e){switch(t){case"focusin":case"focusout":dr=null;break;case"dragenter":case"dragleave":pr=null;break;case"mouseover":case"mouseout":mr=null;break;case"pointerover":case"pointerout":Wl.delete(e.pointerId);break;case"gotpointercapture":case"lostpointercapture":Xl.delete(e.pointerId)}}function ul(t,e,n,i,s,r){return t===null||t.nativeEvent!==r?(t={blockedOn:e,domEventName:n,eventSystemFlags:i,nativeEvent:r,targetContainers:[s]},e!==null&&(e=So(e),e!==null&&RA(e)),t):(t.eventSystemFlags|=i,e=t.targetContainers,s!==null&&e.indexOf(s)===-1&&e.push(s),t)}function fw(t,e,n,i,s){switch(e){case"focusin":return dr=ul(dr,t,e,n,i,s),!0;case"dragenter":return pr=ul(pr,t,e,n,i,s),!0;case"mouseover":return mr=ul(mr,t,e,n,i,s),!0;case"pointerover":var r=s.pointerId;return Wl.set(r,ul(Wl.get(r)||null,t,e,n,i,s)),!0;case"gotpointercapture":return r=s.pointerId,Xl.set(r,ul(Xl.get(r)||null,t,e,n,i,s)),!0}return!1}function UA(t){var e=qr(t.target);if(e!==null){var n=Yl(e);if(n!==null){if(e=n.tag,e===13){if(e=Ny(n),e!==null){t.blockedOn=e,ex(t.priority,function(){wy(n)});return}}else if(e===31){if(e=Py(n),e!==null){t.blockedOn=e,ex(t.priority,function(){wy(n)});return}}else if(e===3&&n.stateNode.current.memoizedState.isDehydrated){t.blockedOn=n.tag===3?n.stateNode.containerInfo:null;return}}}t.blockedOn=null}function Uu(t){if(t.blockedOn!==null)return!1;for(var e=t.targetContainers;0<e.length;){var n=Cm(t.nativeEvent);if(n===null){n=t.nativeEvent;var i=new n.constructor(n.type,n);Ip=i,n.target.dispatchEvent(i),Ip=null}else return e=So(n),e!==null&&RA(e),t.blockedOn=n,!1;e.shift()}return!0}function Ry(t,e,n){Uu(t)&&n.delete(e)}function hw(){Rm=!1,dr!==null&&Uu(dr)&&(dr=null),pr!==null&&Uu(pr)&&(pr=null),mr!==null&&Uu(mr)&&(mr=null),Wl.forEach(Ry),Xl.forEach(Ry)}function fu(t,e){t.blockedOn===e&&(t.blockedOn=null,Rm||(Rm=!0,sn.unstable_scheduleCallback(sn.unstable_NormalPriority,hw)))}var hu=null;function Dy(t){hu!==t&&(hu=t,sn.unstable_scheduleCallback(sn.unstable_NormalPriority,function(){hu===t&&(hu=null);for(var e=0;e<t.length;e+=3){var n=t[e],i=t[e+1],s=t[e+2];if(typeof i!="function"){if(Cg(i||n)===null)continue;break}var r=So(n);r!==null&&(t.splice(e,3),e-=3,Qp(r,{pending:!0,data:s,method:n.method,action:i},i,s))}}))}function yo(t){function e(l){return fu(l,t)}dr!==null&&fu(dr,t),pr!==null&&fu(pr,t),mr!==null&&fu(mr,t),Wl.forEach(e),Xl.forEach(e);for(var n=0;n<$s.length;n++){var i=$s[n];i.blockedOn===t&&(i.blockedOn=null)}for(;0<$s.length&&(n=$s[0],n.blockedOn===null);)UA(n),n.blockedOn===null&&$s.shift();if(n=(t.ownerDocument||t).$$reactFormReplay,n!=null)for(i=0;i<n.length;i+=3){var s=n[i],r=n[i+1],a=s[qn]||null;if(typeof r=="function")a||Dy(n);else if(a){var o=null;if(r&&r.hasAttribute("formAction")){if(s=r,a=r[qn]||null)o=a.formAction;else if(Cg(s)!==null)continue}else o=a.action;typeof o=="function"?n[i+1]=o:(n.splice(i,3),i-=3),Dy(n)}}}function IA(){function t(r){r.canIntercept&&r.info==="react-transition"&&r.intercept({handler:function(){return new Promise(function(a){return s=a})},focusReset:"manual",scroll:"manual"})}function e(){s!==null&&(s(),s=null),i||setTimeout(n,20)}function n(){if(!i&&!navigation.transition){var r=navigation.currentEntry;r&&r.url!=null&&navigation.navigate(r.url,{state:r.getState(),info:"react-transition",history:"replace"})}}if(typeof navigation=="object"){var i=!1,s=null;return navigation.addEventListener("navigate",t),navigation.addEventListener("navigatesuccess",e),navigation.addEventListener("navigateerror",e),setTimeout(n,100),function(){i=!0,navigation.removeEventListener("navigate",t),navigation.removeEventListener("navigatesuccess",e),navigation.removeEventListener("navigateerror",e),s!==null&&(s(),s=null)}}}function Rg(t){this._internalRoot=t}bf.prototype.render=Rg.prototype.render=function(t){var e=this._internalRoot;if(e===null)throw Error(J(409));var n=e.current,i=ri();CA(n,i,t,e,null,null)};bf.prototype.unmount=Rg.prototype.unmount=function(){var t=this._internalRoot;if(t!==null){this._internalRoot=null;var e=t.containerInfo;CA(t.current,2,null,t,null,null),Mf(),e[_o]=null}};function bf(t){this._internalRoot=t}bf.prototype.unstable_scheduleHydration=function(t){if(t){var e=Zy();t={blockedOn:null,target:t,priority:e};for(var n=0;n<$s.length&&e!==0&&e<$s[n].priority;n++);$s.splice(n,0,t),n===0&&UA(t)}};var Uy=Iy.version;if(Uy!=="19.3.0")throw Error(J(527,Uy,"19.3.0"));rt.findDOMNode=function(t){var e=t._reactInternals;if(e===void 0)throw typeof t.render=="function"?Error(J(188)):(t=Object.keys(t).join(","),Error(J(268,t)));return t=dE(e),t=t!==null?Ly(t):null,t=t===null?null:t.stateNode,t};var dw={bundleType:0,version:"19.3.0",rendererPackageName:"react-dom",currentDispatcherRef:Ue,reconcilerVersion:"19.3.0"};if(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__<"u"&&(fl=__REACT_DEVTOOLS_GLOBAL_HOOK__,!fl.isDisabled&&fl.supportsFiber))try{ql=fl.inject(dw),ii=fl}catch{}var fl;wf.createRoot=function(t,e){if(!By(t))throw Error(J(299));var n=!1,i="",s=hS,r=dS,a=pS;return e!=null&&(e.unstable_strictMode===!0&&(n=!0),e.identifierPrefix!==void 0&&(i=e.identifierPrefix),e.onUncaughtError!==void 0&&(s=e.onUncaughtError),e.onCaughtError!==void 0&&(r=e.onCaughtError),e.onRecoverableError!==void 0&&(a=e.onRecoverableError)),e=bA(t,1,!1,null,null,n,i,null,s,r,a,IA),t[_o]=e.current,_g(t),new Rg(e)};wf.hydrateRoot=function(t,e,n){if(!By(t))throw Error(J(299));var i=!1,s="",r=hS,a=dS,o=pS,l=null;return n!=null&&(n.unstable_strictMode===!0&&(i=!0),n.identifierPrefix!==void 0&&(s=n.identifierPrefix),n.onUncaughtError!==void 0&&(r=n.onUncaughtError),n.onCaughtError!==void 0&&(a=n.onCaughtError),n.onRecoverableError!==void 0&&(o=n.onRecoverableError),n.formState!==void 0&&(l=n.formState)),e=bA(t,1,!0,e,n??null,i,s,l,r,a,o,IA),e.context=wA(null),n=e.current,i=ri(),i=Bm(i),s=lr(i),s.callback=null,cr(n,s,i),n=i,e.current.lanes=n,Zl(e,n),cs(e),t[_o]=e.current,_g(t),new bf(e)};wf.version="19.3.0"});var LA=Qi((K3,PA)=>{"use strict";function NA(){if(!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__>"u"||typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE!="function"))try{__REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(NA)}catch(t){console.error(t)}}NA(),PA.exports=BA()});var bM=Qi(Ad=>{"use strict";var N3=Symbol.for("react.transitional.element"),P3=Symbol.for("react.fragment");function TM(t,e,n){var i=null;if(n!==void 0&&(i=""+n),e.key!==void 0&&(i=""+e.key),"key"in e){n={};for(var s in e)s!=="key"&&(n[s]=e[s])}else n=e;return e=n.ref,{$$typeof:N3,type:t,key:i,ref:e!==void 0?e:null,props:n}}Ad.Fragment=P3;Ad.jsx=TM;Ad.jsxs=TM});var Fc=Qi((wL,wM)=>{"use strict";wM.exports=bM()});var Ed=Gr(tl()),DM=Gr(LA());var n1=0,s0=1,i1=2;var Ec=1,s1=2,Qo=3,ps=0,$t=1,zn=2,Kn=0,Zo=1,r0=2,a0=3,o0=4,r1=5;var _a=100,a1=101,o1=102,l1=103,c1=104,u1=200,f1=201,h1=202,d1=203,l0=204,c0=205,p1=206,m1=207,g1=208,v1=209,x1=210,y1=211,_1=212,S1=213,A1=214,Zf=0,Go=1,Kf=2,Vo=3,Jf=4,jf=5,$f=6,eh=7,u0=0,M1=1,E1=2,ki=0,f0=1,h0=2,d0=3,p0=4,m0=5,g0=6,v0=7;var x0=300,Pr=301,Sa=302,wh=303,Ch=304,Tc=306,th=1e3,fs=1001,nh=1002,dn=1003,T1=1004;var bc=1005;var wt=1006,Rh=1007;var Lr=1008;var Xt=1009,y0=1010,_0=1011,Ko=1012,Dh=1013,Wi=1014,fi=1015,Xi=1016,Uh=1017,Ih=1018,Or=1020,S0=35902,A0=35899,M0=1021,E0=1022,Ri=1023,hs=1026,ms=1027,T0=1028,Bh=1029,Fr=1030,Nh=1031;var Ph=1033,wc=33776,Cc=33777,Rc=33778,Dc=33779,Lh=35840,Oh=35841,Fh=35842,zh=35843,Hh=36196,Gh=37492,Vh=37496,kh=37488,Wh=37489,Uc=37490,Xh=37491,Yh=37808,qh=37809,Qh=37810,Zh=37811,Kh=37812,Jh=37813,jh=37814,$h=37815,ed=37816,td=37817,nd=37818,id=37819,sd=37820,rd=37821,ad=36492,od=36494,ld=36495,cd=36283,ud=36284,Ic=36285,fd=36286;var cc=2300,ih=2301,qf=2302,jg=2303,$g=2400,e0=2401,t0=2402;var gs=3200;var b0=0,b1=1,hi="",Dt="srgb",Gs="srgb-linear",uc="linear",ct="srgb";var Qf=7680;var w1=519,C1=512,R1=513,D1=514,hd=515,U1=516,I1=517,dd=518,B1=519,N1=35044;var Bc="300 es",Gi=2e3,fc=2001;function pw(t){for(let e=t.length-1;e>=0;--e)if(t[e]>=65535)return!0;return!1}function mw(t){return ArrayBuffer.isView(t)&&!(t instanceof DataView)}function hc(t){return document.createElementNS("http://www.w3.org/1999/xhtml",t)}function P1(){let t=hc("canvas");return t.style.display="block",t}var OA={},ko=null;function w0(...t){let e="THREE."+t.shift();ko?ko("log",e,...t):console.log(e,...t)}function L1(t){let e=t[0];if(typeof e=="string"&&e.startsWith("TSL:")){let n=t[1];n&&n.isStackTrace?t[0]+=" "+n.getLocation():t[1]='Stack trace not available. Enable "THREE.Node.captureStackTrace" to capture stack traces.'}return t}function De(...t){t=L1(t);let e="THREE."+t.shift();if(ko)ko("warn",e,...t);else{let n=t[0];n&&n.isStackTrace?console.warn(n.getError(e)):console.warn(e,...t)}}function Ie(...t){t=L1(t);let e="THREE."+t.shift();if(ko)ko("error",e,...t);else{let n=t[0];n&&n.isStackTrace?console.error(n.getError(e)):console.error(e,...t)}}function xa(...t){let e=t.join(" ");e in OA||(OA[e]=!0,De(...t))}function O1(t,e,n){return new Promise(function(i,s){function r(){switch(t.clientWaitSync(e,t.SYNC_FLUSH_COMMANDS_BIT,0)){case t.WAIT_FAILED:s();break;case t.TIMEOUT_EXPIRED:setTimeout(r,n);break;default:i()}}setTimeout(r,n)})}var F1={[Zf]:Go,[Kf]:$f,[Jf]:eh,[Vo]:jf,[Go]:Zf,[$f]:Kf,[eh]:Jf,[jf]:Vo},Zn=class{addEventListener(e,n){this._listeners===void 0&&(this._listeners={});let i=this._listeners;i[e]===void 0&&(i[e]=[]),i[e].indexOf(n)===-1&&i[e].push(n)}hasEventListener(e,n){let i=this._listeners;return i===void 0?!1:i[e]!==void 0&&i[e].indexOf(n)!==-1}removeEventListener(e,n){let i=this._listeners;if(i===void 0)return;let s=i[e];if(s!==void 0){let r=s.indexOf(n);r!==-1&&s.splice(r,1)}}dispatchEvent(e){let n=this._listeners;if(n===void 0)return;let i=n[e.type];if(i!==void 0){e.target=this;let s=i.slice(0);for(let r=0,a=s.length;r<a;r++)s[r].call(this,e);e.target=null}}},bn=["00","01","02","03","04","05","06","07","08","09","0a","0b","0c","0d","0e","0f","10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f","20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f","30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f","40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f","50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f","60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f","70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f","80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f","90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f","a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af","b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf","d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df","e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff"];var Dg=Math.PI/180,sh=180/Math.PI;function Nc(){let t=Math.random()*4294967295|0,e=Math.random()*4294967295|0,n=Math.random()*4294967295|0,i=Math.random()*4294967295|0;return(bn[t&255]+bn[t>>8&255]+bn[t>>16&255]+bn[t>>24&255]+"-"+bn[e&255]+bn[e>>8&255]+"-"+bn[e>>16&15|64]+bn[e>>24&255]+"-"+bn[n&63|128]+bn[n>>8&255]+"-"+bn[n>>16&255]+bn[n>>24&255]+bn[i&255]+bn[i>>8&255]+bn[i>>16&255]+bn[i>>24&255]).toLowerCase()}function Je(t,e,n){return Math.max(e,Math.min(n,t))}function gw(t,e){return(t%e+e)%e}function Ug(t,e,n){return(1-n)*t+n*e}function sc(t,e){switch(e.constructor){case Float32Array:return t;case Uint32Array:return t/4294967295;case Uint16Array:return t/65535;case Uint8Array:case Uint8ClampedArray:return t/255;case Int32Array:return Math.max(t/2147483647,-1);case Int16Array:return Math.max(t/32767,-1);case Int8Array:return Math.max(t/127,-1);default:throw new Error("THREE.MathUtils: Invalid component type.")}}function Qn(t,e){switch(e.constructor){case Float32Array:return t;case Uint32Array:return Math.round(t*4294967295);case Uint16Array:return Math.round(t*65535);case Uint8Array:case Uint8ClampedArray:return Math.round(t*255);case Int32Array:return Math.round(t*2147483647);case Int16Array:return Math.round(t*32767);case Int8Array:return Math.round(t*127);default:throw new Error("THREE.MathUtils: Invalid component type.")}}var I0=class I0{constructor(e=0,n=0){this.x=e,this.y=n}get width(){return this.x}set width(e){this.x=e}get height(){return this.y}set height(e){this.y=e}set(e,n){return this.x=e,this.y=n,this}setScalar(e){return this.x=e,this.y=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setComponent(e,n){switch(e){case 0:this.x=n;break;case 1:this.y=n;break;default:throw new Error("THREE.Vector2: index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;default:throw new Error("THREE.Vector2: index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y)}copy(e){return this.x=e.x,this.y=e.y,this}add(e){return this.x+=e.x,this.y+=e.y,this}addScalar(e){return this.x+=e,this.y+=e,this}addVectors(e,n){return this.x=e.x+n.x,this.y=e.y+n.y,this}addScaledVector(e,n){return this.x+=e.x*n,this.y+=e.y*n,this}sub(e){return this.x-=e.x,this.y-=e.y,this}subScalar(e){return this.x-=e,this.y-=e,this}subVectors(e,n){return this.x=e.x-n.x,this.y=e.y-n.y,this}multiply(e){return this.x*=e.x,this.y*=e.y,this}multiplyScalar(e){return this.x*=e,this.y*=e,this}divide(e){return this.x/=e.x,this.y/=e.y,this}divideScalar(e){return this.multiplyScalar(1/e)}applyMatrix3(e){let n=this.x,i=this.y,s=e.elements;return this.x=s[0]*n+s[3]*i+s[6],this.y=s[1]*n+s[4]*i+s[7],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this}clamp(e,n){return this.x=Je(this.x,e.x,n.x),this.y=Je(this.y,e.y,n.y),this}clampScalar(e,n){return this.x=Je(this.x,e,n),this.y=Je(this.y,e,n),this}clampLength(e,n){let i=this.length();return this.divideScalar(i||1).multiplyScalar(Je(i,e,n))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this}negate(){return this.x=-this.x,this.y=-this.y,this}dot(e){return this.x*e.x+this.y*e.y}cross(e){return this.x*e.y-this.y*e.x}lengthSq(){return this.x*this.x+this.y*this.y}length(){return Math.sqrt(this.x*this.x+this.y*this.y)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)}normalize(){return this.divideScalar(this.length()||1)}angle(){return Math.atan2(-this.y,-this.x)+Math.PI}angleTo(e){let n=Math.sqrt(this.lengthSq()*e.lengthSq());if(n===0)return Math.PI/2;let i=this.dot(e)/n;return Math.acos(Je(i,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){let n=this.x-e.x,i=this.y-e.y;return n*n+i*i}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,n){return this.x+=(e.x-this.x)*n,this.y+=(e.y-this.y)*n,this}lerpVectors(e,n,i){return this.x=e.x+(n.x-e.x)*i,this.y=e.y+(n.y-e.y)*i,this}equals(e){return e.x===this.x&&e.y===this.y}fromArray(e,n=0){return this.x=e[n],this.y=e[n+1],this}toArray(e=[],n=0){return e[n]=this.x,e[n+1]=this.y,e}fromBufferAttribute(e,n){return this.x=e.getX(n),this.y=e.getY(n),this}rotateAround(e,n){let i=Math.cos(n),s=Math.sin(n),r=this.x-e.x,a=this.y-e.y;return this.x=r*i-a*s+e.x,this.y=r*s+a*i+e.y,this}random(){return this.x=Math.random(),this.y=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y}};I0.prototype.isVector2=!0;var Ne=I0,ds=class{constructor(e=0,n=0,i=0,s=1){this.isQuaternion=!0,this._x=e,this._y=n,this._z=i,this._w=s}static slerpFlat(e,n,i,s,r,a,o){let l=i[s+0],c=i[s+1],h=i[s+2],p=i[s+3],u=r[a+0],d=r[a+1],v=r[a+2],M=r[a+3];if(p!==M||l!==u||c!==d||h!==v){let m=l*u+c*d+h*v+p*M;m<0&&(u=-u,d=-d,v=-v,M=-M,m=-m);let f=1-o;if(m<.9995){let g=Math.acos(m),S=Math.sin(g);f=Math.sin(f*g)/S,o=Math.sin(o*g)/S,l=l*f+u*o,c=c*f+d*o,h=h*f+v*o,p=p*f+M*o}else{l=l*f+u*o,c=c*f+d*o,h=h*f+v*o,p=p*f+M*o;let g=1/Math.sqrt(l*l+c*c+h*h+p*p);l*=g,c*=g,h*=g,p*=g}}e[n]=l,e[n+1]=c,e[n+2]=h,e[n+3]=p}static multiplyQuaternionsFlat(e,n,i,s,r,a){let o=i[s],l=i[s+1],c=i[s+2],h=i[s+3],p=r[a],u=r[a+1],d=r[a+2],v=r[a+3];return e[n]=o*v+h*p+l*d-c*u,e[n+1]=l*v+h*u+c*p-o*d,e[n+2]=c*v+h*d+o*u-l*p,e[n+3]=h*v-o*p-l*u-c*d,e}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get w(){return this._w}set w(e){this._w=e,this._onChangeCallback()}set(e,n,i,s){return this._x=e,this._y=n,this._z=i,this._w=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._w)}copy(e){return this._x=e.x,this._y=e.y,this._z=e.z,this._w=e.w,this._onChangeCallback(),this}setFromEuler(e,n=!0){let i=e._x,s=e._y,r=e._z,a=e._order,o=Math.cos,l=Math.sin,c=o(i/2),h=o(s/2),p=o(r/2),u=l(i/2),d=l(s/2),v=l(r/2);switch(a){case"XYZ":this._x=u*h*p+c*d*v,this._y=c*d*p-u*h*v,this._z=c*h*v+u*d*p,this._w=c*h*p-u*d*v;break;case"YXZ":this._x=u*h*p+c*d*v,this._y=c*d*p-u*h*v,this._z=c*h*v-u*d*p,this._w=c*h*p+u*d*v;break;case"ZXY":this._x=u*h*p-c*d*v,this._y=c*d*p+u*h*v,this._z=c*h*v+u*d*p,this._w=c*h*p-u*d*v;break;case"ZYX":this._x=u*h*p-c*d*v,this._y=c*d*p+u*h*v,this._z=c*h*v-u*d*p,this._w=c*h*p+u*d*v;break;case"YZX":this._x=u*h*p+c*d*v,this._y=c*d*p+u*h*v,this._z=c*h*v-u*d*p,this._w=c*h*p-u*d*v;break;case"XZY":this._x=u*h*p-c*d*v,this._y=c*d*p-u*h*v,this._z=c*h*v+u*d*p,this._w=c*h*p+u*d*v;break;default:De("Quaternion: .setFromEuler() encountered an unknown order: "+a)}return n===!0&&this._onChangeCallback(),this}setFromAxisAngle(e,n){let i=n/2,s=Math.sin(i);return this._x=e.x*s,this._y=e.y*s,this._z=e.z*s,this._w=Math.cos(i),this._onChangeCallback(),this}setFromRotationMatrix(e){let n=e.elements,i=n[0],s=n[4],r=n[8],a=n[1],o=n[5],l=n[9],c=n[2],h=n[6],p=n[10],u=i+o+p;if(u>0){let d=.5/Math.sqrt(u+1);this._w=.25/d,this._x=(h-l)*d,this._y=(r-c)*d,this._z=(a-s)*d}else if(i>o&&i>p){let d=2*Math.sqrt(1+i-o-p);this._w=(h-l)/d,this._x=.25*d,this._y=(s+a)/d,this._z=(r+c)/d}else if(o>p){let d=2*Math.sqrt(1+o-i-p);this._w=(r-c)/d,this._x=(s+a)/d,this._y=.25*d,this._z=(l+h)/d}else{let d=2*Math.sqrt(1+p-i-o);this._w=(a-s)/d,this._x=(r+c)/d,this._y=(l+h)/d,this._z=.25*d}return this._onChangeCallback(),this}setFromUnitVectors(e,n){let i=e.dot(n)+1;return i<1e-8?(i=0,Math.abs(e.x)>Math.abs(e.z)?(this._x=-e.y,this._y=e.x,this._z=0,this._w=i):(this._x=0,this._y=-e.z,this._z=e.y,this._w=i)):(this._x=e.y*n.z-e.z*n.y,this._y=e.z*n.x-e.x*n.z,this._z=e.x*n.y-e.y*n.x,this._w=i),this.normalize()}angleTo(e){return 2*Math.acos(Math.abs(Je(this.dot(e),-1,1)))}rotateTowards(e,n){let i=this.angleTo(e);if(i===0)return this;let s=Math.min(1,n/i);return this.slerp(e,s),this}identity(){return this.set(0,0,0,1)}invert(){return this.conjugate()}conjugate(){return this._x*=-1,this._y*=-1,this._z*=-1,this._onChangeCallback(),this}dot(e){return this._x*e._x+this._y*e._y+this._z*e._z+this._w*e._w}lengthSq(){return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w}length(){return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)}normalize(){let e=this.length();return e===0?(this._x=0,this._y=0,this._z=0,this._w=1):(e=1/e,this._x=this._x*e,this._y=this._y*e,this._z=this._z*e,this._w=this._w*e),this._onChangeCallback(),this}multiply(e){return this.multiplyQuaternions(this,e)}premultiply(e){return this.multiplyQuaternions(e,this)}multiplyQuaternions(e,n){let i=e._x,s=e._y,r=e._z,a=e._w,o=n._x,l=n._y,c=n._z,h=n._w;return this._x=i*h+a*o+s*c-r*l,this._y=s*h+a*l+r*o-i*c,this._z=r*h+a*c+i*l-s*o,this._w=a*h-i*o-s*l-r*c,this._onChangeCallback(),this}slerp(e,n){let i=e._x,s=e._y,r=e._z,a=e._w,o=this.dot(e);o<0&&(i=-i,s=-s,r=-r,a=-a,o=-o);let l=1-n;if(o<.9995){let c=Math.acos(o),h=Math.sin(c);l=Math.sin(l*c)/h,n=Math.sin(n*c)/h,this._x=this._x*l+i*n,this._y=this._y*l+s*n,this._z=this._z*l+r*n,this._w=this._w*l+a*n,this._onChangeCallback()}else this._x=this._x*l+i*n,this._y=this._y*l+s*n,this._z=this._z*l+r*n,this._w=this._w*l+a*n,this.normalize();return this}slerpQuaternions(e,n,i){return this.copy(e).slerp(n,i)}random(){let e=2*Math.PI*Math.random(),n=2*Math.PI*Math.random(),i=Math.random(),s=Math.sqrt(1-i),r=Math.sqrt(i);return this.set(s*Math.sin(e),s*Math.cos(e),r*Math.sin(n),r*Math.cos(n))}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._w===this._w}fromArray(e,n=0){return this._x=e[n],this._y=e[n+1],this._z=e[n+2],this._w=e[n+3],this._onChangeCallback(),this}toArray(e=[],n=0){return e[n]=this._x,e[n+1]=this._y,e[n+2]=this._z,e[n+3]=this._w,e}fromBufferAttribute(e,n){return this._x=e.getX(n),this._y=e.getY(n),this._z=e.getZ(n),this._w=e.getW(n),this._onChangeCallback(),this}toJSON(){return this.toArray()}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._w}},B0=class B0{constructor(e=0,n=0,i=0){this.x=e,this.y=n,this.z=i}set(e,n,i){return i===void 0&&(i=this.z),this.x=e,this.y=n,this.z=i,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setComponent(e,n){switch(e){case 0:this.x=n;break;case 1:this.y=n;break;case 2:this.z=n;break;default:throw new Error("THREE.Vector3: index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;default:throw new Error("THREE.Vector3: index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this}addVectors(e,n){return this.x=e.x+n.x,this.y=e.y+n.y,this.z=e.z+n.z,this}addScaledVector(e,n){return this.x+=e.x*n,this.y+=e.y*n,this.z+=e.z*n,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this}subVectors(e,n){return this.x=e.x-n.x,this.y=e.y-n.y,this.z=e.z-n.z,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this}multiplyVectors(e,n){return this.x=e.x*n.x,this.y=e.y*n.y,this.z=e.z*n.z,this}applyEuler(e){return this.applyQuaternion(FA.setFromEuler(e))}applyAxisAngle(e,n){return this.applyQuaternion(FA.setFromAxisAngle(e,n))}applyMatrix3(e){let n=this.x,i=this.y,s=this.z,r=e.elements;return this.x=r[0]*n+r[3]*i+r[6]*s,this.y=r[1]*n+r[4]*i+r[7]*s,this.z=r[2]*n+r[5]*i+r[8]*s,this}applyNormalMatrix(e){return this.applyMatrix3(e).normalize()}applyMatrix4(e){let n=this.x,i=this.y,s=this.z,r=e.elements,a=1/(r[3]*n+r[7]*i+r[11]*s+r[15]);return this.x=(r[0]*n+r[4]*i+r[8]*s+r[12])*a,this.y=(r[1]*n+r[5]*i+r[9]*s+r[13])*a,this.z=(r[2]*n+r[6]*i+r[10]*s+r[14])*a,this}applyQuaternion(e){let n=this.x,i=this.y,s=this.z,r=e.x,a=e.y,o=e.z,l=e.w,c=2*(a*s-o*i),h=2*(o*n-r*s),p=2*(r*i-a*n);return this.x=n+l*c+a*p-o*h,this.y=i+l*h+o*c-r*p,this.z=s+l*p+r*h-a*c,this}project(e){return this.applyMatrix4(e.matrixWorldInverse).applyMatrix4(e.projectionMatrix)}unproject(e){return this.applyMatrix4(e.projectionMatrixInverse).applyMatrix4(e.matrixWorld)}transformDirection(e){let n=this.x,i=this.y,s=this.z,r=e.elements;return this.x=r[0]*n+r[4]*i+r[8]*s,this.y=r[1]*n+r[5]*i+r[9]*s,this.z=r[2]*n+r[6]*i+r[10]*s,this.normalize()}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this}divideScalar(e){return this.multiplyScalar(1/e)}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this}clamp(e,n){return this.x=Je(this.x,e.x,n.x),this.y=Je(this.y,e.y,n.y),this.z=Je(this.z,e.z,n.z),this}clampScalar(e,n){return this.x=Je(this.x,e,n),this.y=Je(this.y,e,n),this.z=Je(this.z,e,n),this}clampLength(e,n){let i=this.length();return this.divideScalar(i||1).multiplyScalar(Je(i,e,n))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,n){return this.x+=(e.x-this.x)*n,this.y+=(e.y-this.y)*n,this.z+=(e.z-this.z)*n,this}lerpVectors(e,n,i){return this.x=e.x+(n.x-e.x)*i,this.y=e.y+(n.y-e.y)*i,this.z=e.z+(n.z-e.z)*i,this}cross(e){return this.crossVectors(this,e)}crossVectors(e,n){let i=e.x,s=e.y,r=e.z,a=n.x,o=n.y,l=n.z;return this.x=s*l-r*o,this.y=r*a-i*l,this.z=i*o-s*a,this}projectOnVector(e){let n=e.lengthSq();if(n===0)return this.set(0,0,0);let i=e.dot(this)/n;return this.copy(e).multiplyScalar(i)}projectOnPlane(e){return Ig.copy(this).projectOnVector(e),this.sub(Ig)}reflect(e){return this.sub(Ig.copy(e).multiplyScalar(2*this.dot(e)))}angleTo(e){let n=Math.sqrt(this.lengthSq()*e.lengthSq());if(n===0)return Math.PI/2;let i=this.dot(e)/n;return Math.acos(Je(i,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){let n=this.x-e.x,i=this.y-e.y,s=this.z-e.z;return n*n+i*i+s*s}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)+Math.abs(this.z-e.z)}setFromSpherical(e){return this.setFromSphericalCoords(e.radius,e.phi,e.theta)}setFromSphericalCoords(e,n,i){let s=Math.sin(n)*e;return this.x=s*Math.sin(i),this.y=Math.cos(n)*e,this.z=s*Math.cos(i),this}setFromCylindrical(e){return this.setFromCylindricalCoords(e.radius,e.theta,e.y)}setFromCylindricalCoords(e,n,i){return this.x=e*Math.sin(n),this.y=i,this.z=e*Math.cos(n),this}setFromMatrixPosition(e){let n=e.elements;return this.x=n[12],this.y=n[13],this.z=n[14],this}setFromMatrixScale(e){let n=this.setFromMatrixColumn(e,0).length(),i=this.setFromMatrixColumn(e,1).length(),s=this.setFromMatrixColumn(e,2).length();return this.x=n,this.y=i,this.z=s,this}setFromMatrixColumn(e,n){return this.fromArray(e.elements,n*4)}setFromMatrix3Column(e,n){return this.fromArray(e.elements,n*3)}setFromEuler(e){return this.x=e._x,this.y=e._y,this.z=e._z,this}setFromColor(e){return this.x=e.r,this.y=e.g,this.z=e.b,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z}fromArray(e,n=0){return this.x=e[n],this.y=e[n+1],this.z=e[n+2],this}toArray(e=[],n=0){return e[n]=this.x,e[n+1]=this.y,e[n+2]=this.z,e}fromBufferAttribute(e,n){return this.x=e.getX(n),this.y=e.getY(n),this.z=e.getZ(n),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this}randomDirection(){let e=Math.random()*Math.PI*2,n=Math.random()*2-1,i=Math.sqrt(1-n*n);return this.x=i*Math.cos(e),this.y=n,this.z=i*Math.sin(e),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z}};B0.prototype.isVector3=!0;var G=B0,Ig=new G,FA=new ds,N0=class N0{constructor(e,n,i,s,r,a,o,l,c){this.elements=[1,0,0,0,1,0,0,0,1],e!==void 0&&this.set(e,n,i,s,r,a,o,l,c)}set(e,n,i,s,r,a,o,l,c){let h=this.elements;return h[0]=e,h[1]=s,h[2]=o,h[3]=n,h[4]=r,h[5]=l,h[6]=i,h[7]=a,h[8]=c,this}identity(){return this.set(1,0,0,0,1,0,0,0,1),this}copy(e){let n=this.elements,i=e.elements;return n[0]=i[0],n[1]=i[1],n[2]=i[2],n[3]=i[3],n[4]=i[4],n[5]=i[5],n[6]=i[6],n[7]=i[7],n[8]=i[8],this}extractBasis(e,n,i){return e.setFromMatrix3Column(this,0),n.setFromMatrix3Column(this,1),i.setFromMatrix3Column(this,2),this}setFromMatrix4(e){let n=e.elements;return this.set(n[0],n[4],n[8],n[1],n[5],n[9],n[2],n[6],n[10]),this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,n){let i=e.elements,s=n.elements,r=this.elements,a=i[0],o=i[3],l=i[6],c=i[1],h=i[4],p=i[7],u=i[2],d=i[5],v=i[8],M=s[0],m=s[3],f=s[6],g=s[1],S=s[4],_=s[7],E=s[2],T=s[5],C=s[8];return r[0]=a*M+o*g+l*E,r[3]=a*m+o*S+l*T,r[6]=a*f+o*_+l*C,r[1]=c*M+h*g+p*E,r[4]=c*m+h*S+p*T,r[7]=c*f+h*_+p*C,r[2]=u*M+d*g+v*E,r[5]=u*m+d*S+v*T,r[8]=u*f+d*_+v*C,this}multiplyScalar(e){let n=this.elements;return n[0]*=e,n[3]*=e,n[6]*=e,n[1]*=e,n[4]*=e,n[7]*=e,n[2]*=e,n[5]*=e,n[8]*=e,this}determinant(){let e=this.elements,n=e[0],i=e[1],s=e[2],r=e[3],a=e[4],o=e[5],l=e[6],c=e[7],h=e[8];return n*a*h-n*o*c-i*r*h+i*o*l+s*r*c-s*a*l}invert(){let e=this.elements,n=e[0],i=e[1],s=e[2],r=e[3],a=e[4],o=e[5],l=e[6],c=e[7],h=e[8],p=h*a-o*c,u=o*l-h*r,d=c*r-a*l,v=n*p+i*u+s*d;if(v===0)return this.set(0,0,0,0,0,0,0,0,0);let M=1/v;return e[0]=p*M,e[1]=(s*c-h*i)*M,e[2]=(o*i-s*a)*M,e[3]=u*M,e[4]=(h*n-s*l)*M,e[5]=(s*r-o*n)*M,e[6]=d*M,e[7]=(i*l-c*n)*M,e[8]=(a*n-i*r)*M,this}transpose(){let e,n=this.elements;return e=n[1],n[1]=n[3],n[3]=e,e=n[2],n[2]=n[6],n[6]=e,e=n[5],n[5]=n[7],n[7]=e,this}getNormalMatrix(e){return this.setFromMatrix4(e).invert().transpose()}transposeIntoArray(e){let n=this.elements;return e[0]=n[0],e[1]=n[3],e[2]=n[6],e[3]=n[1],e[4]=n[4],e[5]=n[7],e[6]=n[2],e[7]=n[5],e[8]=n[8],this}setUvTransform(e,n,i,s,r,a,o){let l=Math.cos(r),c=Math.sin(r);return this.set(i*l,i*c,-i*(l*a+c*o)+a+e,-s*c,s*l,-s*(-c*a+l*o)+o+n,0,0,1),this}scale(e,n){return xa("Matrix3: .scale() is deprecated. Use .makeScale() instead."),this.premultiply(Bg.makeScale(e,n)),this}rotate(e){return xa("Matrix3: .rotate() is deprecated. Use .makeRotation() instead."),this.premultiply(Bg.makeRotation(-e)),this}translate(e,n){return xa("Matrix3: .translate() is deprecated. Use .makeTranslation() instead."),this.premultiply(Bg.makeTranslation(e,n)),this}makeTranslation(e,n){return e.isVector2?this.set(1,0,e.x,0,1,e.y,0,0,1):this.set(1,0,e,0,1,n,0,0,1),this}makeRotation(e){let n=Math.cos(e),i=Math.sin(e);return this.set(n,-i,0,i,n,0,0,0,1),this}makeScale(e,n){return this.set(e,0,0,0,n,0,0,0,1),this}equals(e){let n=this.elements,i=e.elements;for(let s=0;s<9;s++)if(n[s]!==i[s])return!1;return!0}fromArray(e,n=0){for(let i=0;i<9;i++)this.elements[i]=e[i+n];return this}toArray(e=[],n=0){let i=this.elements;return e[n]=i[0],e[n+1]=i[1],e[n+2]=i[2],e[n+3]=i[3],e[n+4]=i[4],e[n+5]=i[5],e[n+6]=i[6],e[n+7]=i[7],e[n+8]=i[8],e}clone(){return new this.constructor().fromArray(this.elements)}};N0.prototype.isMatrix3=!0;var Pe=N0,Bg=new Pe,zA=new Pe().set(.4123908,.3575843,.1804808,.212639,.7151687,.0721923,.0193308,.1191948,.9505322),HA=new Pe().set(3.2409699,-1.5373832,-.4986108,-.9692436,1.8759675,.0415551,.0556301,-.203977,1.0569715);function vw(){let t={enabled:!0,workingColorSpace:Gs,spaces:{},convert:function(s,r,a){return this.enabled===!1||r===a||!r||!a||(this.spaces[r].transfer===ct&&(s.r=Hs(s.r),s.g=Hs(s.g),s.b=Hs(s.b)),this.spaces[r].primaries!==this.spaces[a].primaries&&(s.applyMatrix3(this.spaces[r].toXYZ),s.applyMatrix3(this.spaces[a].fromXYZ)),this.spaces[a].transfer===ct&&(s.r=Ho(s.r),s.g=Ho(s.g),s.b=Ho(s.b))),s},workingToColorSpace:function(s,r){return this.convert(s,this.workingColorSpace,r)},colorSpaceToWorking:function(s,r){return this.convert(s,r,this.workingColorSpace)},getPrimaries:function(s){return this.spaces[s].primaries},getTransfer:function(s){return s===hi?uc:this.spaces[s].transfer},getToneMappingMode:function(s){return this.spaces[s].outputColorSpaceConfig.toneMappingMode||"standard"},getLuminanceCoefficients:function(s,r=this.workingColorSpace){return s.fromArray(this.spaces[r].luminanceCoefficients)},define:function(s){Object.assign(this.spaces,s)},_getMatrix:function(s,r,a){return s.copy(this.spaces[r].toXYZ).multiply(this.spaces[a].fromXYZ)},_getDrawingBufferColorSpace:function(s){return this.spaces[s].outputColorSpaceConfig.drawingBufferColorSpace},_getUnpackColorSpace:function(s=this.workingColorSpace){return this.spaces[s].workingColorSpaceConfig.unpackColorSpace},fromWorkingColorSpace:function(s,r){return xa("ColorManagement: .fromWorkingColorSpace() has been renamed to .workingToColorSpace()."),t.workingToColorSpace(s,r)},toWorkingColorSpace:function(s,r){return xa("ColorManagement: .toWorkingColorSpace() has been renamed to .colorSpaceToWorking()."),t.colorSpaceToWorking(s,r)}},e=[.64,.33,.3,.6,.15,.06],n=[.2126,.7152,.0722],i=[.3127,.329];return t.define({[Gs]:{primaries:e,whitePoint:i,transfer:uc,toXYZ:zA,fromXYZ:HA,luminanceCoefficients:n,workingColorSpaceConfig:{unpackColorSpace:Dt},outputColorSpaceConfig:{drawingBufferColorSpace:Dt}},[Dt]:{primaries:e,whitePoint:i,transfer:ct,toXYZ:zA,fromXYZ:HA,luminanceCoefficients:n,outputColorSpaceConfig:{drawingBufferColorSpace:Dt}}}),t}var Ke=vw();function Hs(t){return t<.04045?t*.0773993808:Math.pow(t*.9478672986+.0521327014,2.4)}function Ho(t){return t<.0031308?t*12.92:1.055*Math.pow(t,.41666)-.055}var wo,rh=class{static getDataURL(e,n="image/png"){if(/^data:/i.test(e.src)||typeof HTMLCanvasElement>"u")return e.src;let i;if(e instanceof HTMLCanvasElement)i=e;else{wo===void 0&&(wo=hc("canvas")),wo.width=e.width,wo.height=e.height;let s=wo.getContext("2d");e instanceof ImageData?s.putImageData(e,0,0):s.drawImage(e,0,0,e.width,e.height),i=wo}return i.toDataURL(n)}static sRGBToLinear(e){if(typeof HTMLImageElement<"u"&&e instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&e instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&e instanceof ImageBitmap){let n=hc("canvas");n.width=e.width,n.height=e.height;let i=n.getContext("2d");i.drawImage(e,0,0,e.width,e.height);let s=i.getImageData(0,0,e.width,e.height),r=s.data;for(let a=0;a<r.length;a++)r[a]=Hs(r[a]/255)*255;return i.putImageData(s,0,0),n}else if(e.data){let n=e.data.slice(0);for(let i=0;i<n.length;i++)n instanceof Uint8Array||n instanceof Uint8ClampedArray?n[i]=Math.floor(Hs(n[i]/255)*255):n[i]=Hs(n[i]);return{data:n,width:e.width,height:e.height}}else return De("ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."),e}},xw=0,Wo=class{constructor(e=null){this.isTextureSource=!0,Object.defineProperty(this,"id",{value:xw++}),this.uuid=Nc(),this.data=e,this.dataReady=!0,this.version=0}getSize(e){let n=this.data;return typeof HTMLVideoElement<"u"&&n instanceof HTMLVideoElement?e.set(n.videoWidth,n.videoHeight,0):typeof VideoFrame<"u"&&n instanceof VideoFrame?e.set(n.displayWidth,n.displayHeight,0):n!==null?e.set(n.width,n.height,n.depth||0):e.set(0,0,0),e}set needsUpdate(e){e===!0&&this.version++}toJSON(e){let n=e===void 0||typeof e=="string";if(!n&&e.images[this.uuid]!==void 0)return e.images[this.uuid];let i={uuid:this.uuid,url:""},s=this.data;if(s!==null){let r;if(Array.isArray(s)){r=[];for(let a=0,o=s.length;a<o;a++)s[a].isDataTexture?r.push(Ng(s[a].image)):r.push(Ng(s[a]))}else r=Ng(s);i.url=r}return n||(e.images[this.uuid]=i),i}};function Ng(t){return typeof HTMLImageElement<"u"&&t instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&t instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&t instanceof ImageBitmap?rh.getDataURL(t):t.data?{data:Array.from(t.data),width:t.width,height:t.height,type:t.data.constructor.name}:(De("Texture: Unable to serialize Texture."),{})}var yw=0,Pg=new G,jt=class t extends Zn{constructor(e=t.DEFAULT_IMAGE,n=t.DEFAULT_MAPPING,i=fs,s=fs,r=wt,a=Lr,o=Ri,l=Xt,c=t.DEFAULT_ANISOTROPY,h=hi){super(),this.isTexture=!0,Object.defineProperty(this,"id",{value:yw++}),this.uuid=Nc(),this.name="",this.source=new Wo(e),this.mipmaps=[],this.mapping=n,this.channel=0,this.wrapS=i,this.wrapT=s,this.magFilter=r,this.minFilter=a,this.anisotropy=c,this.format=o,this.internalFormat=null,this.type=l,this.offset=new Ne(0,0),this.repeat=new Ne(1,1),this.center=new Ne(0,0),this.rotation=0,this.matrixAutoUpdate=!0,this.matrix=new Pe,this.generateMipmaps=!0,this.premultiplyAlpha=!1,this.flipY=!0,this.unpackAlignment=4,this.colorSpace=h,this.userData={},this.updateRanges=[],this.version=0,this.onUpdate=null,this.renderTarget=null,this.isRenderTargetTexture=!1,this.isArrayTexture=!!(e&&e.depth&&e.depth>1),this.pmremVersion=0,this.normalized=!1}get width(){return this.source.getSize(Pg).x}get height(){return this.source.getSize(Pg).y}get depth(){return this.source.getSize(Pg).z}get image(){return this.source.data}set image(e){this.source.data=e}updateMatrix(){this.matrix.setUvTransform(this.offset.x,this.offset.y,this.repeat.x,this.repeat.y,this.rotation,this.center.x,this.center.y)}addUpdateRange(e,n){this.updateRanges.push({start:e,count:n})}clearUpdateRanges(){this.updateRanges.length=0}clone(){return new this.constructor().copy(this)}copy(e){return this.name=e.name,this.source=e.source,this.mipmaps=e.mipmaps.slice(0),this.mapping=e.mapping,this.channel=e.channel,this.wrapS=e.wrapS,this.wrapT=e.wrapT,this.magFilter=e.magFilter,this.minFilter=e.minFilter,this.anisotropy=e.anisotropy,this.format=e.format,this.internalFormat=e.internalFormat,this.type=e.type,this.normalized=e.normalized,this.offset.copy(e.offset),this.repeat.copy(e.repeat),this.center.copy(e.center),this.rotation=e.rotation,this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrix.copy(e.matrix),this.generateMipmaps=e.generateMipmaps,this.premultiplyAlpha=e.premultiplyAlpha,this.flipY=e.flipY,this.unpackAlignment=e.unpackAlignment,this.colorSpace=e.colorSpace,this.renderTarget=e.renderTarget,this.isRenderTargetTexture=e.isRenderTargetTexture,this.isArrayTexture=e.isArrayTexture,this.userData=JSON.parse(JSON.stringify(e.userData)),this.needsUpdate=!0,this}setValues(e){for(let n in e){let i=e[n];if(i===void 0){De(`Texture.setValues(): parameter '${n}' has value of undefined.`);continue}let s=this[n];if(s===void 0){De(`Texture.setValues(): property '${n}' does not exist.`);continue}s&&i&&s.isVector2&&i.isVector2||s&&i&&s.isVector3&&i.isVector3||s&&i&&s.isMatrix3&&i.isMatrix3?s.copy(i):this[n]=i}}toJSON(e){let n=e===void 0||typeof e=="string";if(!n&&e.textures[this.uuid]!==void 0)return e.textures[this.uuid];let i={metadata:{version:4.7,type:"Texture",generator:"Texture.toJSON"},uuid:this.uuid,name:this.name,image:this.source.toJSON(e).uuid,mapping:this.mapping,channel:this.channel,repeat:[this.repeat.x,this.repeat.y],offset:[this.offset.x,this.offset.y],center:[this.center.x,this.center.y],rotation:this.rotation,wrap:[this.wrapS,this.wrapT],format:this.format,internalFormat:this.internalFormat,type:this.type,normalized:this.normalized,colorSpace:this.colorSpace,minFilter:this.minFilter,magFilter:this.magFilter,anisotropy:this.anisotropy,flipY:this.flipY,generateMipmaps:this.generateMipmaps,premultiplyAlpha:this.premultiplyAlpha,unpackAlignment:this.unpackAlignment};return Object.keys(this.userData).length>0&&(i.userData=this.userData),n||(e.textures[this.uuid]=i),i}dispose(){this.dispatchEvent({type:"dispose"})}transformUv(e){if(this.mapping!==x0)return e;if(e.applyMatrix3(this.matrix),e.x<0||e.x>1)switch(this.wrapS){case th:e.x=e.x-Math.floor(e.x);break;case fs:e.x=e.x<0?0:1;break;case nh:Math.abs(Math.floor(e.x)%2)===1?e.x=Math.ceil(e.x)-e.x:e.x=e.x-Math.floor(e.x);break}if(e.y<0||e.y>1)switch(this.wrapT){case th:e.y=e.y-Math.floor(e.y);break;case fs:e.y=e.y<0?0:1;break;case nh:Math.abs(Math.floor(e.y)%2)===1?e.y=Math.ceil(e.y)-e.y:e.y=e.y-Math.floor(e.y);break}return this.flipY&&(e.y=1-e.y),e}set needsUpdate(e){e===!0&&(this.version++,this.source.needsUpdate=!0)}set needsPMREMUpdate(e){e===!0&&this.pmremVersion++}};jt.DEFAULT_IMAGE=null;jt.DEFAULT_MAPPING=x0;jt.DEFAULT_ANISOTROPY=1;var P0=class P0{constructor(e=0,n=0,i=0,s=1){this.x=e,this.y=n,this.z=i,this.w=s}get width(){return this.z}set width(e){this.z=e}get height(){return this.w}set height(e){this.w=e}set(e,n,i,s){return this.x=e,this.y=n,this.z=i,this.w=s,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this.w=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setW(e){return this.w=e,this}setComponent(e,n){switch(e){case 0:this.x=n;break;case 1:this.y=n;break;case 2:this.z=n;break;case 3:this.w=n;break;default:throw new Error("THREE.Vector4: index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;case 3:return this.w;default:throw new Error("THREE.Vector4: index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z,this.w)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this.w=e.w!==void 0?e.w:1,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this.w+=e.w,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this.w+=e,this}addVectors(e,n){return this.x=e.x+n.x,this.y=e.y+n.y,this.z=e.z+n.z,this.w=e.w+n.w,this}addScaledVector(e,n){return this.x+=e.x*n,this.y+=e.y*n,this.z+=e.z*n,this.w+=e.w*n,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this.w-=e.w,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this.w-=e,this}subVectors(e,n){return this.x=e.x-n.x,this.y=e.y-n.y,this.z=e.z-n.z,this.w=e.w-n.w,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this.w*=e.w,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this.w*=e,this}applyMatrix4(e){let n=this.x,i=this.y,s=this.z,r=this.w,a=e.elements;return this.x=a[0]*n+a[4]*i+a[8]*s+a[12]*r,this.y=a[1]*n+a[5]*i+a[9]*s+a[13]*r,this.z=a[2]*n+a[6]*i+a[10]*s+a[14]*r,this.w=a[3]*n+a[7]*i+a[11]*s+a[15]*r,this}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this.w/=e.w,this}divideScalar(e){return this.multiplyScalar(1/e)}setAxisAngleFromQuaternion(e){this.w=2*Math.acos(e.w);let n=Math.sqrt(1-e.w*e.w);return n<1e-4?(this.x=1,this.y=0,this.z=0):(this.x=e.x/n,this.y=e.y/n,this.z=e.z/n),this}setAxisAngleFromRotationMatrix(e){let n,i,s,r,l=e.elements,c=l[0],h=l[4],p=l[8],u=l[1],d=l[5],v=l[9],M=l[2],m=l[6],f=l[10];if(Math.abs(h-u)<.01&&Math.abs(p-M)<.01&&Math.abs(v-m)<.01){if(Math.abs(h+u)<.1&&Math.abs(p+M)<.1&&Math.abs(v+m)<.1&&Math.abs(c+d+f-3)<.1)return this.set(1,0,0,0),this;n=Math.PI;let S=(c+1)/2,_=(d+1)/2,E=(f+1)/2,T=(h+u)/4,C=(p+M)/4,y=(v+m)/4;return S>_&&S>E?S<.01?(i=0,s=.707106781,r=.707106781):(i=Math.sqrt(S),s=T/i,r=C/i):_>E?_<.01?(i=.707106781,s=0,r=.707106781):(s=Math.sqrt(_),i=T/s,r=y/s):E<.01?(i=.707106781,s=.707106781,r=0):(r=Math.sqrt(E),i=C/r,s=y/r),this.set(i,s,r,n),this}let g=Math.sqrt((m-v)*(m-v)+(p-M)*(p-M)+(u-h)*(u-h));return Math.abs(g)<.001&&(g=1),this.x=(m-v)/g,this.y=(p-M)/g,this.z=(u-h)/g,this.w=Math.acos((c+d+f-1)/2),this}setFromMatrixPosition(e){let n=e.elements;return this.x=n[12],this.y=n[13],this.z=n[14],this.w=n[15],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this.w=Math.min(this.w,e.w),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this.w=Math.max(this.w,e.w),this}clamp(e,n){return this.x=Je(this.x,e.x,n.x),this.y=Je(this.y,e.y,n.y),this.z=Je(this.z,e.z,n.z),this.w=Je(this.w,e.w,n.w),this}clampScalar(e,n){return this.x=Je(this.x,e,n),this.y=Je(this.y,e,n),this.z=Je(this.z,e,n),this.w=Je(this.w,e,n),this}clampLength(e,n){let i=this.length();return this.divideScalar(i||1).multiplyScalar(Je(i,e,n))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this.w=Math.floor(this.w),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this.w=Math.ceil(this.w),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this.w=Math.round(this.w),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this.w=Math.trunc(this.w),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this.w=-this.w,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z+this.w*e.w}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,n){return this.x+=(e.x-this.x)*n,this.y+=(e.y-this.y)*n,this.z+=(e.z-this.z)*n,this.w+=(e.w-this.w)*n,this}lerpVectors(e,n,i){return this.x=e.x+(n.x-e.x)*i,this.y=e.y+(n.y-e.y)*i,this.z=e.z+(n.z-e.z)*i,this.w=e.w+(n.w-e.w)*i,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z&&e.w===this.w}fromArray(e,n=0){return this.x=e[n],this.y=e[n+1],this.z=e[n+2],this.w=e[n+3],this}toArray(e=[],n=0){return e[n]=this.x,e[n+1]=this.y,e[n+2]=this.z,e[n+3]=this.w,e}fromBufferAttribute(e,n){return this.x=e.getX(n),this.y=e.getY(n),this.z=e.getZ(n),this.w=e.getW(n),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this.w=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z,yield this.w}};P0.prototype.isVector4=!0;var Ot=P0,ah=class extends Zn{constructor(e=1,n=1,i={}){super(),i=Object.assign({generateMipmaps:!1,internalFormat:null,minFilter:wt,depthBuffer:!0,stencilBuffer:!1,resolveColorBuffer:!0,resolveDepthBuffer:!0,resolveStencilBuffer:!0,storeMultisampledColorBuffer:!0,storeMultisampledDepthBuffer:!0,storeMultisampledStencilBuffer:!0,depthTexture:null,samples:0,count:1,depth:1,multiview:!1,useArrayDepthTexture:!1},i),this.isRenderTarget=!0,this.width=e,this.height=n,this.depth=i.depth,this.scissor=new Ot(0,0,e,n),this.scissorTest=!1,this.viewport=new Ot(0,0,e,n),this.textures=[];let s={width:e,height:n,depth:i.depth},r=new jt(s),a=i.count;for(let o=0;o<a;o++)this.textures[o]=r.clone(),this.textures[o].isRenderTargetTexture=!0,this.textures[o].renderTarget=this;this._setTextureOptions(i),this.depthBuffer=i.depthBuffer,this.stencilBuffer=i.stencilBuffer,this.resolveColorBuffer=i.resolveColorBuffer,this.resolveDepthBuffer=i.resolveDepthBuffer,this.resolveStencilBuffer=i.resolveStencilBuffer,this.storeMultisampledColorBuffer=i.storeMultisampledColorBuffer,this.storeMultisampledDepthBuffer=i.storeMultisampledDepthBuffer,this.storeMultisampledStencilBuffer=i.storeMultisampledStencilBuffer,this._depthTexture=null,this.depthTexture=i.depthTexture,this.samples=i.samples,this.multiview=i.multiview,this.useArrayDepthTexture=i.useArrayDepthTexture}_setTextureOptions(e={}){let n={minFilter:wt,generateMipmaps:!1,flipY:!1,internalFormat:null};e.mapping!==void 0&&(n.mapping=e.mapping),e.wrapS!==void 0&&(n.wrapS=e.wrapS),e.wrapT!==void 0&&(n.wrapT=e.wrapT),e.wrapR!==void 0&&(n.wrapR=e.wrapR),e.magFilter!==void 0&&(n.magFilter=e.magFilter),e.minFilter!==void 0&&(n.minFilter=e.minFilter),e.format!==void 0&&(n.format=e.format),e.type!==void 0&&(n.type=e.type),e.anisotropy!==void 0&&(n.anisotropy=e.anisotropy),e.colorSpace!==void 0&&(n.colorSpace=e.colorSpace),e.flipY!==void 0&&(n.flipY=e.flipY),e.generateMipmaps!==void 0&&(n.generateMipmaps=e.generateMipmaps),e.internalFormat!==void 0&&(n.internalFormat=e.internalFormat);for(let i=0;i<this.textures.length;i++)this.textures[i].setValues(n)}get texture(){return this.textures[0]}set texture(e){this.textures[0]=e}set depthTexture(e){this._depthTexture!==null&&this._depthTexture.renderTarget===this&&(this._depthTexture.renderTarget=null),e!==null&&e.renderTarget===null&&(e.renderTarget=this),this._depthTexture=e}get depthTexture(){return this._depthTexture}setSize(e,n,i=1){if(this.width!==e||this.height!==n||this.depth!==i){this.width=e,this.height=n,this.depth=i;for(let s=0,r=this.textures.length;s<r;s++)this.textures[s].image.width=e,this.textures[s].image.height=n,this.textures[s].image.depth=i,this.textures[s].isData3DTexture!==!0&&(this.textures[s].isArrayTexture=this.textures[s].image.depth>1);this.dispose()}this.viewport.set(0,0,e,n),this.scissor.set(0,0,e,n)}clone(){return new this.constructor().copy(this)}copy(e){this.width=e.width,this.height=e.height,this.depth=e.depth,this.scissor.copy(e.scissor),this.scissorTest=e.scissorTest,this.viewport.copy(e.viewport),this.textures.length=0;for(let n=0,i=e.textures.length;n<i;n++){this.textures[n]=e.textures[n].clone(),this.textures[n].isRenderTargetTexture=!0,this.textures[n].renderTarget=this;let s=Object.assign({},e.textures[n].image);this.textures[n].source=new Wo(s)}if(this.depthBuffer=e.depthBuffer,this.stencilBuffer=e.stencilBuffer,this.resolveColorBuffer=e.resolveColorBuffer,this.resolveDepthBuffer=e.resolveDepthBuffer,this.resolveStencilBuffer=e.resolveStencilBuffer,this.storeMultisampledColorBuffer=e.storeMultisampledColorBuffer,this.storeMultisampledDepthBuffer=e.storeMultisampledDepthBuffer,this.storeMultisampledStencilBuffer=e.storeMultisampledStencilBuffer,e.depthTexture!==null)if(e.depthTexture.renderTarget===e){let n=e.depthTexture.clone();n.renderTarget=null,this.depthTexture=n}else this.depthTexture=e.depthTexture;return this.samples=e.samples,this.multiview=e.multiview,this.useArrayDepthTexture=e.useArrayDepthTexture,this}dispose(){this.dispatchEvent({type:"dispose"})}},Ft=class extends ah{constructor(e=1,n=1,i={}){super(e,n,i),this.isWebGLRenderTarget=!0}},dc=class extends jt{constructor(e=null,n=1,i=1,s=1){super(null),this.isDataArrayTexture=!0,this.image={data:e,width:n,height:i,depth:s},this.magFilter=dn,this.minFilter=dn,this.wrapR=fs,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1,this.layerUpdates=new Set}copy(e){return super.copy(e),this.wrapR=e.wrapR,this}addLayerUpdate(e){this.layerUpdates.add(e)}clearLayerUpdates(){this.layerUpdates.clear()}};var oh=class extends jt{constructor(e=null,n=1,i=1,s=1){super(null),this.isData3DTexture=!0,this.image={data:e,width:n,height:i,depth:s},this.magFilter=dn,this.minFilter=dn,this.wrapR=fs,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}copy(e){return super.copy(e),this.wrapR=e.wrapR,this}};var bh=class bh{constructor(e,n,i,s,r,a,o,l,c,h,p,u,d,v,M,m){this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],e!==void 0&&this.set(e,n,i,s,r,a,o,l,c,h,p,u,d,v,M,m)}set(e,n,i,s,r,a,o,l,c,h,p,u,d,v,M,m){let f=this.elements;return f[0]=e,f[4]=n,f[8]=i,f[12]=s,f[1]=r,f[5]=a,f[9]=o,f[13]=l,f[2]=c,f[6]=h,f[10]=p,f[14]=u,f[3]=d,f[7]=v,f[11]=M,f[15]=m,this}identity(){return this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),this}clone(){return new bh().fromArray(this.elements)}copy(e){let n=this.elements,i=e.elements;return n[0]=i[0],n[1]=i[1],n[2]=i[2],n[3]=i[3],n[4]=i[4],n[5]=i[5],n[6]=i[6],n[7]=i[7],n[8]=i[8],n[9]=i[9],n[10]=i[10],n[11]=i[11],n[12]=i[12],n[13]=i[13],n[14]=i[14],n[15]=i[15],this}copyPosition(e){let n=this.elements,i=e.elements;return n[12]=i[12],n[13]=i[13],n[14]=i[14],this}setFromMatrix3(e){let n=e.elements;return this.set(n[0],n[3],n[6],0,n[1],n[4],n[7],0,n[2],n[5],n[8],0,0,0,0,1),this}extractBasis(e,n,i){return this.determinantAffine()===0?(e.set(1,0,0),n.set(0,1,0),i.set(0,0,1),this):(e.setFromMatrixColumn(this,0),n.setFromMatrixColumn(this,1),i.setFromMatrixColumn(this,2),this)}makeBasis(e,n,i){return this.set(e.x,n.x,i.x,0,e.y,n.y,i.y,0,e.z,n.z,i.z,0,0,0,0,1),this}extractRotation(e){if(e.determinantAffine()===0)return this.identity();let n=this.elements,i=e.elements,s=1/Co.setFromMatrixColumn(e,0).length(),r=1/Co.setFromMatrixColumn(e,1).length(),a=1/Co.setFromMatrixColumn(e,2).length();return n[0]=i[0]*s,n[1]=i[1]*s,n[2]=i[2]*s,n[3]=0,n[4]=i[4]*r,n[5]=i[5]*r,n[6]=i[6]*r,n[7]=0,n[8]=i[8]*a,n[9]=i[9]*a,n[10]=i[10]*a,n[11]=0,n[12]=0,n[13]=0,n[14]=0,n[15]=1,this}makeRotationFromEuler(e){let n=this.elements,i=e.x,s=e.y,r=e.z,a=Math.cos(i),o=Math.sin(i),l=Math.cos(s),c=Math.sin(s),h=Math.cos(r),p=Math.sin(r);if(e.order==="XYZ"){let u=a*h,d=a*p,v=o*h,M=o*p;n[0]=l*h,n[4]=-l*p,n[8]=c,n[1]=d+v*c,n[5]=u-M*c,n[9]=-o*l,n[2]=M-u*c,n[6]=v+d*c,n[10]=a*l}else if(e.order==="YXZ"){let u=l*h,d=l*p,v=c*h,M=c*p;n[0]=u+M*o,n[4]=v*o-d,n[8]=a*c,n[1]=a*p,n[5]=a*h,n[9]=-o,n[2]=d*o-v,n[6]=M+u*o,n[10]=a*l}else if(e.order==="ZXY"){let u=l*h,d=l*p,v=c*h,M=c*p;n[0]=u-M*o,n[4]=-a*p,n[8]=v+d*o,n[1]=d+v*o,n[5]=a*h,n[9]=M-u*o,n[2]=-a*c,n[6]=o,n[10]=a*l}else if(e.order==="ZYX"){let u=a*h,d=a*p,v=o*h,M=o*p;n[0]=l*h,n[4]=v*c-d,n[8]=u*c+M,n[1]=l*p,n[5]=M*c+u,n[9]=d*c-v,n[2]=-c,n[6]=o*l,n[10]=a*l}else if(e.order==="YZX"){let u=a*l,d=a*c,v=o*l,M=o*c;n[0]=l*h,n[4]=M-u*p,n[8]=v*p+d,n[1]=p,n[5]=a*h,n[9]=-o*h,n[2]=-c*h,n[6]=d*p+v,n[10]=u-M*p}else if(e.order==="XZY"){let u=a*l,d=a*c,v=o*l,M=o*c;n[0]=l*h,n[4]=-p,n[8]=c*h,n[1]=u*p+M,n[5]=a*h,n[9]=d*p-v,n[2]=v*p-d,n[6]=o*h,n[10]=M*p+u}return n[3]=0,n[7]=0,n[11]=0,n[12]=0,n[13]=0,n[14]=0,n[15]=1,this}makeRotationFromQuaternion(e){return this.compose(_w,e,Sw)}lookAt(e,n,i){let s=this.elements;return li.subVectors(e,n),li.lengthSq()===0&&(li.z=1),li.normalize(),Mr.crossVectors(i,li),Mr.lengthSq()===0&&(Math.abs(i.z)===1?li.x+=1e-4:li.z+=1e-4,li.normalize(),Mr.crossVectors(i,li)),Mr.normalize(),Cf.crossVectors(li,Mr),s[0]=Mr.x,s[4]=Cf.x,s[8]=li.x,s[1]=Mr.y,s[5]=Cf.y,s[9]=li.y,s[2]=Mr.z,s[6]=Cf.z,s[10]=li.z,this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,n){let i=e.elements,s=n.elements,r=this.elements,a=i[0],o=i[4],l=i[8],c=i[12],h=i[1],p=i[5],u=i[9],d=i[13],v=i[2],M=i[6],m=i[10],f=i[14],g=i[3],S=i[7],_=i[11],E=i[15],T=s[0],C=s[4],y=s[8],b=s[12],R=s[1],N=s[5],F=s[9],k=s[13],B=s[2],z=s[6],Z=s[10],q=s[14],ie=s[3],W=s[7],$=s[11],te=s[15];return r[0]=a*T+o*R+l*B+c*ie,r[4]=a*C+o*N+l*z+c*W,r[8]=a*y+o*F+l*Z+c*$,r[12]=a*b+o*k+l*q+c*te,r[1]=h*T+p*R+u*B+d*ie,r[5]=h*C+p*N+u*z+d*W,r[9]=h*y+p*F+u*Z+d*$,r[13]=h*b+p*k+u*q+d*te,r[2]=v*T+M*R+m*B+f*ie,r[6]=v*C+M*N+m*z+f*W,r[10]=v*y+M*F+m*Z+f*$,r[14]=v*b+M*k+m*q+f*te,r[3]=g*T+S*R+_*B+E*ie,r[7]=g*C+S*N+_*z+E*W,r[11]=g*y+S*F+_*Z+E*$,r[15]=g*b+S*k+_*q+E*te,this}multiplyScalar(e){let n=this.elements;return n[0]*=e,n[4]*=e,n[8]*=e,n[12]*=e,n[1]*=e,n[5]*=e,n[9]*=e,n[13]*=e,n[2]*=e,n[6]*=e,n[10]*=e,n[14]*=e,n[3]*=e,n[7]*=e,n[11]*=e,n[15]*=e,this}determinant(){let e=this.elements,n=e[0],i=e[4],s=e[8],r=e[12],a=e[1],o=e[5],l=e[9],c=e[13],h=e[2],p=e[6],u=e[10],d=e[14],v=e[3],M=e[7],m=e[11],f=e[15],g=l*d-c*u,S=o*d-c*p,_=o*u-l*p,E=a*d-c*h,T=a*u-l*h,C=a*p-o*h;return n*(M*g-m*S+f*_)-i*(v*g-m*E+f*T)+s*(v*S-M*E+f*C)-r*(v*_-M*T+m*C)}determinantAffine(){let e=this.elements,n=e[0],i=e[4],s=e[8],r=e[1],a=e[5],o=e[9],l=e[2],c=e[6],h=e[10];return n*(a*h-o*c)-i*(r*h-o*l)+s*(r*c-a*l)}transpose(){let e=this.elements,n;return n=e[1],e[1]=e[4],e[4]=n,n=e[2],e[2]=e[8],e[8]=n,n=e[6],e[6]=e[9],e[9]=n,n=e[3],e[3]=e[12],e[12]=n,n=e[7],e[7]=e[13],e[13]=n,n=e[11],e[11]=e[14],e[14]=n,this}setPosition(e,n,i){let s=this.elements;return e.isVector3?(s[12]=e.x,s[13]=e.y,s[14]=e.z):(s[12]=e,s[13]=n,s[14]=i),this}invert(){let e=this.elements,n=e[0],i=e[1],s=e[2],r=e[3],a=e[4],o=e[5],l=e[6],c=e[7],h=e[8],p=e[9],u=e[10],d=e[11],v=e[12],M=e[13],m=e[14],f=e[15],g=n*o-i*a,S=n*l-s*a,_=n*c-r*a,E=i*l-s*o,T=i*c-r*o,C=s*c-r*l,y=h*M-p*v,b=h*m-u*v,R=h*f-d*v,N=p*m-u*M,F=p*f-d*M,k=u*f-d*m,B=g*k-S*F+_*N+E*R-T*b+C*y;if(B===0)return this.set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);let z=1/B;return e[0]=(o*k-l*F+c*N)*z,e[1]=(s*F-i*k-r*N)*z,e[2]=(M*C-m*T+f*E)*z,e[3]=(u*T-p*C-d*E)*z,e[4]=(l*R-a*k-c*b)*z,e[5]=(n*k-s*R+r*b)*z,e[6]=(m*_-v*C-f*S)*z,e[7]=(h*C-u*_+d*S)*z,e[8]=(a*F-o*R+c*y)*z,e[9]=(i*R-n*F-r*y)*z,e[10]=(v*T-M*_+f*g)*z,e[11]=(p*_-h*T-d*g)*z,e[12]=(o*b-a*N-l*y)*z,e[13]=(n*N-i*b+s*y)*z,e[14]=(M*S-v*E-m*g)*z,e[15]=(h*E-p*S+u*g)*z,this}scale(e){let n=this.elements,i=e.x,s=e.y,r=e.z;return n[0]*=i,n[4]*=s,n[8]*=r,n[1]*=i,n[5]*=s,n[9]*=r,n[2]*=i,n[6]*=s,n[10]*=r,n[3]*=i,n[7]*=s,n[11]*=r,this}getMaxScaleOnAxis(){let e=this.elements,n=e[0]*e[0]+e[1]*e[1]+e[2]*e[2],i=e[4]*e[4]+e[5]*e[5]+e[6]*e[6],s=e[8]*e[8]+e[9]*e[9]+e[10]*e[10];return Math.sqrt(Math.max(n,i,s))}makeTranslation(e,n,i){return e.isVector3?this.set(1,0,0,e.x,0,1,0,e.y,0,0,1,e.z,0,0,0,1):this.set(1,0,0,e,0,1,0,n,0,0,1,i,0,0,0,1),this}makeRotationX(e){let n=Math.cos(e),i=Math.sin(e);return this.set(1,0,0,0,0,n,-i,0,0,i,n,0,0,0,0,1),this}makeRotationY(e){let n=Math.cos(e),i=Math.sin(e);return this.set(n,0,i,0,0,1,0,0,-i,0,n,0,0,0,0,1),this}makeRotationZ(e){let n=Math.cos(e),i=Math.sin(e);return this.set(n,-i,0,0,i,n,0,0,0,0,1,0,0,0,0,1),this}makeRotationAxis(e,n){let i=Math.cos(n),s=Math.sin(n),r=1-i,a=e.x,o=e.y,l=e.z,c=r*a,h=r*o;return this.set(c*a+i,c*o-s*l,c*l+s*o,0,c*o+s*l,h*o+i,h*l-s*a,0,c*l-s*o,h*l+s*a,r*l*l+i,0,0,0,0,1),this}makeScale(e,n,i){return this.set(e,0,0,0,0,n,0,0,0,0,i,0,0,0,0,1),this}makeShear(e,n,i,s,r,a){return this.set(1,i,r,0,e,1,a,0,n,s,1,0,0,0,0,1),this}compose(e,n,i){let s=this.elements,r=n._x,a=n._y,o=n._z,l=n._w,c=r+r,h=a+a,p=o+o,u=r*c,d=r*h,v=r*p,M=a*h,m=a*p,f=o*p,g=l*c,S=l*h,_=l*p,E=i.x,T=i.y,C=i.z;return s[0]=(1-(M+f))*E,s[1]=(d+_)*E,s[2]=(v-S)*E,s[3]=0,s[4]=(d-_)*T,s[5]=(1-(u+f))*T,s[6]=(m+g)*T,s[7]=0,s[8]=(v+S)*C,s[9]=(m-g)*C,s[10]=(1-(u+M))*C,s[11]=0,s[12]=e.x,s[13]=e.y,s[14]=e.z,s[15]=1,this}decompose(e,n,i){let s=this.elements;e.x=s[12],e.y=s[13],e.z=s[14];let r=this.determinantAffine();if(r===0)return i.set(1,1,1),n.identity(),this;let a=Co.set(s[0],s[1],s[2]).length(),o=Co.set(s[4],s[5],s[6]).length(),l=Co.set(s[8],s[9],s[10]).length();r<0&&(a=-a),Oi.copy(this);let c=1/a,h=1/o,p=1/l;return Oi.elements[0]*=c,Oi.elements[1]*=c,Oi.elements[2]*=c,Oi.elements[4]*=h,Oi.elements[5]*=h,Oi.elements[6]*=h,Oi.elements[8]*=p,Oi.elements[9]*=p,Oi.elements[10]*=p,n.setFromRotationMatrix(Oi),i.x=a,i.y=o,i.z=l,this}makePerspective(e,n,i,s,r,a,o=Gi,l=!1){let c=this.elements,h=2*r/(n-e),p=2*r/(i-s),u=(n+e)/(n-e),d=(i+s)/(i-s),v,M;if(l)v=r/(a-r),M=a*r/(a-r);else if(o===Gi)v=-(a+r)/(a-r),M=-2*a*r/(a-r);else if(o===fc)v=-a/(a-r),M=-a*r/(a-r);else throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: "+o);return c[0]=h,c[4]=0,c[8]=u,c[12]=0,c[1]=0,c[5]=p,c[9]=d,c[13]=0,c[2]=0,c[6]=0,c[10]=v,c[14]=M,c[3]=0,c[7]=0,c[11]=-1,c[15]=0,this}makeOrthographic(e,n,i,s,r,a,o=Gi,l=!1){let c=this.elements,h=2/(n-e),p=2/(i-s),u=-(n+e)/(n-e),d=-(i+s)/(i-s),v,M;if(l)v=1/(a-r),M=a/(a-r);else if(o===Gi)v=-2/(a-r),M=-(a+r)/(a-r);else if(o===fc)v=-1/(a-r),M=-r/(a-r);else throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: "+o);return c[0]=h,c[4]=0,c[8]=0,c[12]=u,c[1]=0,c[5]=p,c[9]=0,c[13]=d,c[2]=0,c[6]=0,c[10]=v,c[14]=M,c[3]=0,c[7]=0,c[11]=0,c[15]=1,this}equals(e){let n=this.elements,i=e.elements;for(let s=0;s<16;s++)if(n[s]!==i[s])return!1;return!0}fromArray(e,n=0){for(let i=0;i<16;i++)this.elements[i]=e[i+n];return this}toArray(e=[],n=0){let i=this.elements;return e[n]=i[0],e[n+1]=i[1],e[n+2]=i[2],e[n+3]=i[3],e[n+4]=i[4],e[n+5]=i[5],e[n+6]=i[6],e[n+7]=i[7],e[n+8]=i[8],e[n+9]=i[9],e[n+10]=i[10],e[n+11]=i[11],e[n+12]=i[12],e[n+13]=i[13],e[n+14]=i[14],e[n+15]=i[15],e}};bh.prototype.isMatrix4=!0;var kt=bh,Co=new G,Oi=new kt,_w=new G(0,0,0),Sw=new G(1,1,1),Mr=new G,Cf=new G,li=new G,GA=new kt,VA=new ds,Rr=class t{constructor(e=0,n=0,i=0,s=t.DEFAULT_ORDER){this.isEuler=!0,this._x=e,this._y=n,this._z=i,this._order=s}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get order(){return this._order}set order(e){this._order=e,this._onChangeCallback()}set(e,n,i,s=this._order){return this._x=e,this._y=n,this._z=i,this._order=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._order)}copy(e){return this._x=e._x,this._y=e._y,this._z=e._z,this._order=e._order,this._onChangeCallback(),this}setFromRotationMatrix(e,n=this._order,i=!0){let s=e.elements,r=s[0],a=s[4],o=s[8],l=s[1],c=s[5],h=s[9],p=s[2],u=s[6],d=s[10];switch(n){case"XYZ":this._y=Math.asin(Je(o,-1,1)),Math.abs(o)<.9999999?(this._x=Math.atan2(-h,d),this._z=Math.atan2(-a,r)):(this._x=Math.atan2(u,c),this._z=0);break;case"YXZ":this._x=Math.asin(-Je(h,-1,1)),Math.abs(h)<.9999999?(this._y=Math.atan2(o,d),this._z=Math.atan2(l,c)):(this._y=Math.atan2(-p,r),this._z=0);break;case"ZXY":this._x=Math.asin(Je(u,-1,1)),Math.abs(u)<.9999999?(this._y=Math.atan2(-p,d),this._z=Math.atan2(-a,c)):(this._y=0,this._z=Math.atan2(l,r));break;case"ZYX":this._y=Math.asin(-Je(p,-1,1)),Math.abs(p)<.9999999?(this._x=Math.atan2(u,d),this._z=Math.atan2(l,r)):(this._x=0,this._z=Math.atan2(-a,c));break;case"YZX":this._z=Math.asin(Je(l,-1,1)),Math.abs(l)<.9999999?(this._x=Math.atan2(-h,c),this._y=Math.atan2(-p,r)):(this._x=0,this._y=Math.atan2(o,d));break;case"XZY":this._z=Math.asin(-Je(a,-1,1)),Math.abs(a)<.9999999?(this._x=Math.atan2(u,c),this._y=Math.atan2(o,r)):(this._x=Math.atan2(-h,d),this._y=0);break;default:De("Euler: .setFromRotationMatrix() encountered an unknown order: "+n)}return this._order=n,i===!0&&this._onChangeCallback(),this}setFromQuaternion(e,n,i){return GA.makeRotationFromQuaternion(e),this.setFromRotationMatrix(GA,n,i)}setFromVector3(e,n=this._order){return this.set(e.x,e.y,e.z,n)}reorder(e){return VA.setFromEuler(this),this.setFromQuaternion(VA,e)}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._order===this._order}fromArray(e){return this._x=e[0],this._y=e[1],this._z=e[2],e[3]!==void 0&&(this._order=e[3]),this._onChangeCallback(),this}toArray(e=[],n=0){return e[n]=this._x,e[n+1]=this._y,e[n+2]=this._z,e[n+3]=this._order,e}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._order}};Rr.DEFAULT_ORDER="XYZ";var pc=class{constructor(){this.mask=1}set(e){this.mask=(1<<e|0)>>>0}enable(e){this.mask|=1<<e|0}enableAll(){this.mask=-1}toggle(e){this.mask^=1<<e|0}disable(e){this.mask&=~(1<<e|0)}disableAll(){this.mask=0}test(e){return(this.mask&e.mask)!==0}isEnabled(e){return(this.mask&(1<<e|0))!==0}},Aw=0,kA=new G,Ro=new ds,Ps=new kt,Rf=new G,rc=new G,Mw=new G,Ew=new ds,WA=new G(1,0,0),XA=new G(0,1,0),YA=new G(0,0,1),qA={type:"added"},Tw={type:"removed"},Do={type:"childadded",child:null},Lg={type:"childremoved",child:null},bi=class t extends Zn{constructor(){super(),this.isObject3D=!0,Object.defineProperty(this,"id",{value:Aw++}),this.uuid=Nc(),this.name="",this.type="Object3D",this.parent=null,this.children=[],this.up=t.DEFAULT_UP.clone();let e=new G,n=new Rr,i=new ds,s=new G(1,1,1);function r(){i.setFromEuler(n,!1)}function a(){n.setFromQuaternion(i,void 0,!1)}n._onChange(r),i._onChange(a),Object.defineProperties(this,{position:{configurable:!0,enumerable:!0,value:e},rotation:{configurable:!0,enumerable:!0,value:n},quaternion:{configurable:!0,enumerable:!0,value:i},scale:{configurable:!0,enumerable:!0,value:s},modelViewMatrix:{value:new kt},normalMatrix:{value:new Pe}}),this.matrix=new kt,this.matrixWorld=new kt,this.matrixAutoUpdate=t.DEFAULT_MATRIX_AUTO_UPDATE,this.matrixWorldAutoUpdate=t.DEFAULT_MATRIX_WORLD_AUTO_UPDATE,this.matrixWorldNeedsUpdate=!1,this.layers=new pc,this.visible=!0,this.castShadow=!1,this.receiveShadow=!1,this.frustumCulled=!0,this.renderOrder=0,this.animations=[],this.customDepthMaterial=void 0,this.customDistanceMaterial=void 0,this.static=!1,this.userData={},this.pivot=null}onBeforeShadow(){}onAfterShadow(){}onBeforeRender(){}onAfterRender(){}applyMatrix4(e){this.matrixAutoUpdate&&this.updateMatrix(),this.matrix.premultiply(e),this.matrix.decompose(this.position,this.quaternion,this.scale)}applyQuaternion(e){return this.quaternion.premultiply(e),this}setRotationFromAxisAngle(e,n){this.quaternion.setFromAxisAngle(e,n)}setRotationFromEuler(e){this.quaternion.setFromEuler(e,!0)}setRotationFromMatrix(e){this.quaternion.setFromRotationMatrix(e)}setRotationFromQuaternion(e){this.quaternion.copy(e)}rotateOnAxis(e,n){return Ro.setFromAxisAngle(e,n),this.quaternion.multiply(Ro),this}rotateOnWorldAxis(e,n){return Ro.setFromAxisAngle(e,n),this.quaternion.premultiply(Ro),this}rotateX(e){return this.rotateOnAxis(WA,e)}rotateY(e){return this.rotateOnAxis(XA,e)}rotateZ(e){return this.rotateOnAxis(YA,e)}translateOnAxis(e,n){return kA.copy(e).applyQuaternion(this.quaternion),this.position.add(kA.multiplyScalar(n)),this}translateX(e){return this.translateOnAxis(WA,e)}translateY(e){return this.translateOnAxis(XA,e)}translateZ(e){return this.translateOnAxis(YA,e)}localToWorld(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(this.matrixWorld)}worldToLocal(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(Ps.copy(this.matrixWorld).invert())}lookAt(e,n,i){e.isVector3?Rf.copy(e):Rf.set(e,n,i);let s=this.parent;this.updateWorldMatrix(!0,!1),rc.setFromMatrixPosition(this.matrixWorld),this.isCamera||this.isLight?Ps.lookAt(rc,Rf,this.up):Ps.lookAt(Rf,rc,this.up),this.quaternion.setFromRotationMatrix(Ps),s&&(Ps.extractRotation(s.matrixWorld),Ro.setFromRotationMatrix(Ps),this.quaternion.premultiply(Ro.invert()))}add(e){if(arguments.length>1){for(let n=0;n<arguments.length;n++)this.add(arguments[n]);return this}return e===this?(Ie("Object3D.add: object can't be added as a child of itself.",e),this):(e&&e.isObject3D?(e.removeFromParent(),e.parent=this,this.children.push(e),e.dispatchEvent(qA),Do.child=e,this.dispatchEvent(Do),Do.child=null):Ie("Object3D.add: object not an instance of THREE.Object3D.",e),this)}remove(e){if(arguments.length>1){for(let i=0;i<arguments.length;i++)this.remove(arguments[i]);return this}let n=this.children.indexOf(e);return n!==-1&&(e.parent=null,this.children.splice(n,1),e.dispatchEvent(Tw),Lg.child=e,this.dispatchEvent(Lg),Lg.child=null),this}removeFromParent(){let e=this.parent;return e!==null&&e.remove(this),this}clear(){return this.remove(...this.children)}attach(e){return this.updateWorldMatrix(!0,!1),Ps.copy(this.matrixWorld).invert(),e.parent!==null&&(e.parent.updateWorldMatrix(!0,!1),Ps.multiply(e.parent.matrixWorld)),e.applyMatrix4(Ps),e.removeFromParent(),e.parent=this,this.children.push(e),e.updateWorldMatrix(!1,!0),e.dispatchEvent(qA),Do.child=e,this.dispatchEvent(Do),Do.child=null,this}getObjectById(e){return this.getObjectByProperty("id",e)}getObjectByName(e){return this.getObjectByProperty("name",e)}getObjectByProperty(e,n){if(this[e]===n)return this;for(let i=0,s=this.children.length;i<s;i++){let a=this.children[i].getObjectByProperty(e,n);if(a!==void 0)return a}}getObjectsByProperty(e,n,i=[]){this[e]===n&&i.push(this);let s=this.children;for(let r=0,a=s.length;r<a;r++)s[r].getObjectsByProperty(e,n,i);return i}getWorldPosition(e){return this.updateWorldMatrix(!0,!1),e.setFromMatrixPosition(this.matrixWorld)}getWorldQuaternion(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(rc,e,Mw),e}getWorldScale(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(rc,Ew,e),e}getWorldDirection(e){this.updateWorldMatrix(!0,!1);let n=this.matrixWorld.elements;return e.set(n[8],n[9],n[10]).normalize()}raycast(){}intersectsFrustum(){}traverse(e){e(this);let n=this.children;for(let i=0,s=n.length;i<s;i++)n[i].traverse(e)}traverseVisible(e){if(this.visible===!1)return;e(this);let n=this.children;for(let i=0,s=n.length;i<s;i++)n[i].traverseVisible(e)}traverseAncestors(e){let n=this.parent;n!==null&&(e(n),n.traverseAncestors(e))}updateMatrix(){this.matrix.compose(this.position,this.quaternion,this.scale);let e=this.pivot;if(e!==null){let n=e.x,i=e.y,s=e.z,r=this.matrix.elements;r[12]+=n-r[0]*n-r[4]*i-r[8]*s,r[13]+=i-r[1]*n-r[5]*i-r[9]*s,r[14]+=s-r[2]*n-r[6]*i-r[10]*s}this.matrixWorldNeedsUpdate=!0}updateMatrixWorld(e){this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||e)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,e=!0);let n=this.children;for(let i=0,s=n.length;i<s;i++)n[i].updateMatrixWorld(e)}updateWorldMatrix(e,n,i=!1){let s=this.parent;if(e===!0&&s!==null&&s.updateWorldMatrix(!0,!1),this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||i)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,i=!0),n===!0){let r=this.children;for(let a=0,o=r.length;a<o;a++)r[a].updateWorldMatrix(!1,!0,i)}}toJSON(e){let n=e===void 0||typeof e=="string",i={};n&&(e={geometries:{},materials:{},textures:{},images:{},shapes:{},skeletons:{},animations:{},nodes:{}},i.metadata={version:4.7,type:"Object",generator:"Object3D.toJSON"});let s={};s.uuid=this.uuid,s.type=this.type,s.name=this.name,s.castShadow=this.castShadow,s.receiveShadow=this.receiveShadow,s.visible=this.visible,s.frustumCulled=this.frustumCulled,s.renderOrder=this.renderOrder,s.static=this.static,s.matrixAutoUpdate=this.matrixAutoUpdate,Object.keys(this.userData).length>0&&(s.userData=this.userData),s.layers=this.layers.mask,s.matrix=this.matrix.toArray(),s.up=this.up.toArray(),this.pivot!==null&&(s.pivot=this.pivot.toArray()),this.morphTargetDictionary!==void 0&&(s.morphTargetDictionary=Object.assign({},this.morphTargetDictionary)),this.morphTargetInfluences!==void 0&&(s.morphTargetInfluences=this.morphTargetInfluences.slice()),this.isInstancedMesh&&(s.type="InstancedMesh",s.count=this.count,s.instanceMatrix=this.instanceMatrix.toJSON(),this.instanceColor!==null&&(s.instanceColor=this.instanceColor.toJSON())),this.isBatchedMesh&&(s.type="BatchedMesh",s.perObjectFrustumCulled=this.perObjectFrustumCulled,s.sortObjects=this.sortObjects,s.drawRanges=this._drawRanges,s.reservedRanges=this._reservedRanges,s.geometryInfo=this._geometryInfo.map(o=>({...o,boundingBox:o.boundingBox?o.boundingBox.toJSON():void 0,boundingSphere:o.boundingSphere?o.boundingSphere.toJSON():void 0})),s.instanceInfo=this._instanceInfo.map(o=>({...o})),s.availableInstanceIds=this._availableInstanceIds.slice(),s.availableGeometryIds=this._availableGeometryIds.slice(),s.nextIndexStart=this._nextIndexStart,s.nextVertexStart=this._nextVertexStart,s.geometryCount=this._geometryCount,s.maxInstanceCount=this._maxInstanceCount,s.maxVertexCount=this._maxVertexCount,s.maxIndexCount=this._maxIndexCount,s.geometryInitialized=this._geometryInitialized,s.matricesTexture=this._matricesTexture.toJSON(e),s.indirectTexture=this._indirectTexture.toJSON(e),this._colorsTexture!==null&&(s.colorsTexture=this._colorsTexture.toJSON(e)),this.boundingSphere!==null&&(s.boundingSphere=this.boundingSphere.toJSON()),this.boundingBox!==null&&(s.boundingBox=this.boundingBox.toJSON()));function r(o,l){return o[l.uuid]===void 0&&(o[l.uuid]=l.toJSON(e)),l.uuid}if(this.isScene)this.background&&(this.background.isColor?s.background=this.background.toJSON():this.background.isTexture&&(s.background=this.background.toJSON(e).uuid)),this.environment&&this.environment.isTexture&&this.environment.isRenderTargetTexture!==!0&&(s.environment=this.environment.toJSON(e).uuid);else if(this.isMesh||this.isLine||this.isPoints){s.geometry=r(e.geometries,this.geometry);let o=this.geometry.parameters;if(o!==void 0&&o.shapes!==void 0){let l=o.shapes;if(Array.isArray(l))for(let c=0,h=l.length;c<h;c++){let p=l[c];r(e.shapes,p)}else r(e.shapes,l)}}if(this.isSkinnedMesh&&(s.bindMode=this.bindMode,s.bindMatrix=this.bindMatrix.toArray(),this.skeleton!==void 0&&(r(e.skeletons,this.skeleton),s.skeleton=this.skeleton.uuid)),this.material!==void 0)if(Array.isArray(this.material)){let o=[];for(let l=0,c=this.material.length;l<c;l++)o.push(r(e.materials,this.material[l]));s.material=o}else s.material=r(e.materials,this.material);if(this.children.length>0){s.children=[];for(let o=0;o<this.children.length;o++)s.children.push(this.children[o].toJSON(e).object)}if(this.animations.length>0){s.animations=[];for(let o=0;o<this.animations.length;o++){let l=this.animations[o];s.animations.push(r(e.animations,l))}}if(n){let o=a(e.geometries),l=a(e.materials),c=a(e.textures),h=a(e.images),p=a(e.shapes),u=a(e.skeletons),d=a(e.animations),v=a(e.nodes);o.length>0&&(i.geometries=o),l.length>0&&(i.materials=l),c.length>0&&(i.textures=c),h.length>0&&(i.images=h),p.length>0&&(i.shapes=p),u.length>0&&(i.skeletons=u),d.length>0&&(i.animations=d),v.length>0&&(i.nodes=v)}return i.object=s,i;function a(o){let l=[];for(let c in o){let h=o[c];delete h.metadata,l.push(h)}return l}}clone(e){return new this.constructor().copy(this,e)}copy(e,n=!0){if(this.name=e.name,this.up.copy(e.up),this.position.copy(e.position),this.rotation.order=e.rotation.order,this.quaternion.copy(e.quaternion),this.scale.copy(e.scale),this.pivot=e.pivot!==null?e.pivot.clone():null,this.matrix.copy(e.matrix),this.matrixWorld.copy(e.matrixWorld),this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrixWorldAutoUpdate=e.matrixWorldAutoUpdate,this.matrixWorldNeedsUpdate=e.matrixWorldNeedsUpdate,this.layers.mask=e.layers.mask,this.visible=e.visible,this.castShadow=e.castShadow,this.receiveShadow=e.receiveShadow,this.frustumCulled=e.frustumCulled,this.renderOrder=e.renderOrder,this.static=e.static,this.animations=e.animations.slice(),this.userData=JSON.parse(JSON.stringify(e.userData)),n===!0)for(let i=0;i<e.children.length;i++){let s=e.children[i];this.add(s.clone())}return this}dispose(){this.dispatchEvent({type:"dispose"})}};bi.DEFAULT_UP=new G(0,1,0);bi.DEFAULT_MATRIX_AUTO_UPDATE=!0;bi.DEFAULT_MATRIX_WORLD_AUTO_UPDATE=!0;var va=class extends bi{constructor(){super(),this.isGroup=!0,this.type="Group"}},bw={type:"move"},Xo=class{constructor(){this._targetRay=null,this._grip=null,this._hand=null}getHandSpace(){return this._hand===null&&(this._hand=new va,this._hand.matrixAutoUpdate=!1,this._hand.visible=!1,this._hand.joints={},this._hand.inputState={pinching:!1}),this._hand}getTargetRaySpace(){return this._targetRay===null&&(this._targetRay=new va,this._targetRay.matrixAutoUpdate=!1,this._targetRay.visible=!1,this._targetRay.hasLinearVelocity=!1,this._targetRay.linearVelocity=new G,this._targetRay.hasAngularVelocity=!1,this._targetRay.angularVelocity=new G),this._targetRay}getGripSpace(){return this._grip===null&&(this._grip=new va,this._grip.matrixAutoUpdate=!1,this._grip.visible=!1,this._grip.hasLinearVelocity=!1,this._grip.linearVelocity=new G,this._grip.hasAngularVelocity=!1,this._grip.angularVelocity=new G,this._grip.eventsEnabled=!1),this._grip}dispatchEvent(e){return this._targetRay!==null&&this._targetRay.dispatchEvent(e),this._grip!==null&&this._grip.dispatchEvent(e),this._hand!==null&&this._hand.dispatchEvent(e),this}connect(e){if(e&&e.hand){let n=this._hand;if(n)for(let i of e.hand.values())this._getHandJoint(n,i)}return this.dispatchEvent({type:"connected",data:e}),this}disconnect(e){return this.dispatchEvent({type:"disconnected",data:e}),this._targetRay!==null&&(this._targetRay.visible=!1),this._grip!==null&&(this._grip.visible=!1),this._hand!==null&&(this._hand.visible=!1),this}update(e,n,i){let s=null,r=null,a=null,o=this._targetRay,l=this._grip,c=this._hand;if(e&&n.session.visibilityState!=="visible-blurred"){if(c&&e.hand){a=!0;for(let M of e.hand.values()){let m=n.getJointPose(M,i),f=this._getHandJoint(c,M);m!==null&&(f.matrix.fromArray(m.transform.matrix),f.matrix.decompose(f.position,f.rotation,f.scale),f.matrixWorldNeedsUpdate=!0,f.jointRadius=m.radius),f.visible=m!==null}let h=c.joints["index-finger-tip"],p=c.joints["thumb-tip"],u=h.position.distanceTo(p.position),d=.02,v=.005;c.inputState.pinching&&u>d+v?(c.inputState.pinching=!1,this.dispatchEvent({type:"pinchend",handedness:e.handedness,target:this})):!c.inputState.pinching&&u<=d-v&&(c.inputState.pinching=!0,this.dispatchEvent({type:"pinchstart",handedness:e.handedness,target:this}))}else l!==null&&e.gripSpace&&(r=n.getPose(e.gripSpace,i),r!==null&&(l.matrix.fromArray(r.transform.matrix),l.matrix.decompose(l.position,l.rotation,l.scale),l.matrixWorldNeedsUpdate=!0,r.linearVelocity?(l.hasLinearVelocity=!0,l.linearVelocity.copy(r.linearVelocity)):l.hasLinearVelocity=!1,r.angularVelocity?(l.hasAngularVelocity=!0,l.angularVelocity.copy(r.angularVelocity)):l.hasAngularVelocity=!1,l.eventsEnabled&&l.dispatchEvent({type:"gripUpdated",data:e,target:this})));o!==null&&(s=n.getPose(e.targetRaySpace,i),s===null&&r!==null&&(s=r),s!==null&&(o.matrix.fromArray(s.transform.matrix),o.matrix.decompose(o.position,o.rotation,o.scale),o.matrixWorldNeedsUpdate=!0,s.linearVelocity?(o.hasLinearVelocity=!0,o.linearVelocity.copy(s.linearVelocity)):o.hasLinearVelocity=!1,s.angularVelocity?(o.hasAngularVelocity=!0,o.angularVelocity.copy(s.angularVelocity)):o.hasAngularVelocity=!1,this.dispatchEvent(bw)))}return o!==null&&(o.visible=s!==null),l!==null&&(l.visible=r!==null),c!==null&&(c.visible=a!==null),this}_getHandJoint(e,n){if(e.joints[n.jointName]===void 0){let i=new va;i.matrixAutoUpdate=!1,i.visible=!1,e.joints[n.jointName]=i,e.add(i)}return e.joints[n.jointName]}},z1={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074},Er={h:0,s:0,l:0},Df={h:0,s:0,l:0};function Og(t,e,n){return n<0&&(n+=1),n>1&&(n-=1),n<1/6?t+(e-t)*6*n:n<1/2?e:n<2/3?t+(e-t)*6*(2/3-n):t}var Ye=class{constructor(e,n,i){return this.isColor=!0,this.r=1,this.g=1,this.b=1,this.set(e,n,i)}set(e,n,i){if(n===void 0&&i===void 0){let s=e;s&&s.isColor?this.copy(s):typeof s=="number"?this.setHex(s):typeof s=="string"&&this.setStyle(s)}else this.setRGB(e,n,i);return this}setScalar(e){return this.r=e,this.g=e,this.b=e,this}setHex(e,n=Dt){return e=Math.floor(e),this.r=(e>>16&255)/255,this.g=(e>>8&255)/255,this.b=(e&255)/255,Ke.colorSpaceToWorking(this,n),this}setRGB(e,n,i,s=Ke.workingColorSpace){return this.r=e,this.g=n,this.b=i,Ke.colorSpaceToWorking(this,s),this}setHSL(e,n,i,s=Ke.workingColorSpace){if(e=gw(e,1),n=Je(n,0,1),i=Je(i,0,1),n===0)this.r=this.g=this.b=i;else{let r=i<=.5?i*(1+n):i+n-i*n,a=2*i-r;this.r=Og(a,r,e+1/3),this.g=Og(a,r,e),this.b=Og(a,r,e-1/3)}return Ke.colorSpaceToWorking(this,s),this}setStyle(e,n=Dt){function i(r){r!==void 0&&parseFloat(r)<1&&De("Color: Alpha component of "+e+" will be ignored.")}let s;if(s=/^(\w+)\(([^\)]*)\)/.exec(e)){let r,a=s[1],o=s[2];switch(a){case"rgb":case"rgba":if(r=/^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return i(r[4]),this.setRGB(Math.min(255,parseInt(r[1],10))/255,Math.min(255,parseInt(r[2],10))/255,Math.min(255,parseInt(r[3],10))/255,n);if(r=/^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return i(r[4]),this.setRGB(Math.min(100,parseInt(r[1],10))/100,Math.min(100,parseInt(r[2],10))/100,Math.min(100,parseInt(r[3],10))/100,n);break;case"hsl":case"hsla":if(r=/^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return i(r[4]),this.setHSL(parseFloat(r[1])/360,parseFloat(r[2])/100,parseFloat(r[3])/100,n);break;default:De("Color: Unknown color model "+e)}}else if(s=/^\#([A-Fa-f\d]+)$/.exec(e)){let r=s[1],a=r.length;if(a===3)return this.setRGB(parseInt(r.charAt(0),16)/15,parseInt(r.charAt(1),16)/15,parseInt(r.charAt(2),16)/15,n);if(a===6)return this.setHex(parseInt(r,16),n);De("Color: Invalid hex color "+e)}else if(e&&e.length>0)return this.setColorName(e,n);return this}setColorName(e,n=Dt){let i=z1[e.toLowerCase()];return i!==void 0?this.setHex(i,n):De("Color: Unknown color "+e),this}clone(){return new this.constructor(this.r,this.g,this.b)}copy(e){return this.r=e.r,this.g=e.g,this.b=e.b,this}copySRGBToLinear(e){return this.r=Hs(e.r),this.g=Hs(e.g),this.b=Hs(e.b),this}copyLinearToSRGB(e){return this.r=Ho(e.r),this.g=Ho(e.g),this.b=Ho(e.b),this}convertSRGBToLinear(){return this.copySRGBToLinear(this),this}convertLinearToSRGB(){return this.copyLinearToSRGB(this),this}getHex(e=Dt){return Ke.workingToColorSpace(wn.copy(this),e),Math.round(Je(wn.r*255,0,255))*65536+Math.round(Je(wn.g*255,0,255))*256+Math.round(Je(wn.b*255,0,255))}getHexString(e=Dt){return("000000"+this.getHex(e).toString(16)).slice(-6)}getHSL(e,n=Ke.workingColorSpace){Ke.workingToColorSpace(wn.copy(this),n);let i=wn.r,s=wn.g,r=wn.b,a=Math.max(i,s,r),o=Math.min(i,s,r),l,c,h=(o+a)/2;if(o===a)l=0,c=0;else{let p=a-o;switch(c=h<=.5?p/(a+o):p/(2-a-o),a){case i:l=(s-r)/p+(s<r?6:0);break;case s:l=(r-i)/p+2;break;case r:l=(i-s)/p+4;break}l/=6}return e.h=l,e.s=c,e.l=h,e}getRGB(e,n=Ke.workingColorSpace){return Ke.workingToColorSpace(wn.copy(this),n),e.r=wn.r,e.g=wn.g,e.b=wn.b,e}getStyle(e=Dt){Ke.workingToColorSpace(wn.copy(this),e);let n=wn.r,i=wn.g,s=wn.b;return e!==Dt?`color(${e} ${n.toFixed(3)} ${i.toFixed(3)} ${s.toFixed(3)})`:`rgb(${Math.round(n*255)},${Math.round(i*255)},${Math.round(s*255)})`}offsetHSL(e,n,i){return this.getHSL(Er),this.setHSL(Er.h+e,Er.s+n,Er.l+i)}add(e){return this.r+=e.r,this.g+=e.g,this.b+=e.b,this}addColors(e,n){return this.r=e.r+n.r,this.g=e.g+n.g,this.b=e.b+n.b,this}addScalar(e){return this.r+=e,this.g+=e,this.b+=e,this}sub(e){return this.r=Math.max(0,this.r-e.r),this.g=Math.max(0,this.g-e.g),this.b=Math.max(0,this.b-e.b),this}multiply(e){return this.r*=e.r,this.g*=e.g,this.b*=e.b,this}multiplyScalar(e){return this.r*=e,this.g*=e,this.b*=e,this}lerp(e,n){return this.r+=(e.r-this.r)*n,this.g+=(e.g-this.g)*n,this.b+=(e.b-this.b)*n,this}lerpColors(e,n,i){return this.r=e.r+(n.r-e.r)*i,this.g=e.g+(n.g-e.g)*i,this.b=e.b+(n.b-e.b)*i,this}lerpHSL(e,n){this.getHSL(Er),e.getHSL(Df);let i=Ug(Er.h,Df.h,n),s=Ug(Er.s,Df.s,n),r=Ug(Er.l,Df.l,n);return this.setHSL(i,s,r),this}setFromVector3(e){return this.r=e.x,this.g=e.y,this.b=e.z,this}applyMatrix3(e){let n=this.r,i=this.g,s=this.b,r=e.elements;return this.r=r[0]*n+r[3]*i+r[6]*s,this.g=r[1]*n+r[4]*i+r[7]*s,this.b=r[2]*n+r[5]*i+r[8]*s,this}equals(e){return e.r===this.r&&e.g===this.g&&e.b===this.b}fromArray(e,n=0){return this.r=e[n],this.g=e[n+1],this.b=e[n+2],this}toArray(e=[],n=0){return e[n]=this.r,e[n+1]=this.g,e[n+2]=this.b,e}fromBufferAttribute(e,n){return this.r=e.getX(n),this.g=e.getY(n),this.b=e.getZ(n),this}toJSON(){return this.getHex()}*[Symbol.iterator](){yield this.r,yield this.g,yield this.b}},wn=new Ye;Ye.NAMES=z1;var Dr=class extends bi{constructor(){super(),this.isScene=!0,this.type="Scene",this.background=null,this.environment=null,this.fog=null,this.backgroundBlurriness=0,this.backgroundIntensity=1,this.backgroundRotation=new Rr,this.environmentIntensity=1,this.environmentRotation=new Rr,this.overrideMaterial=null,typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}copy(e,n){return super.copy(e,n),e.background!==null&&(this.background=e.background.clone()),e.environment!==null&&(this.environment=e.environment.clone()),e.fog!==null&&(this.fog=e.fog.clone()),this.backgroundBlurriness=e.backgroundBlurriness,this.backgroundIntensity=e.backgroundIntensity,this.backgroundRotation.copy(e.backgroundRotation),this.environmentIntensity=e.environmentIntensity,this.environmentRotation.copy(e.environmentRotation),e.overrideMaterial!==null&&(this.overrideMaterial=e.overrideMaterial.clone()),this.matrixAutoUpdate=e.matrixAutoUpdate,this}toJSON(e){let n=super.toJSON(e);return this.fog!==null&&(n.object.fog=this.fog.toJSON()),n.object.backgroundBlurriness=this.backgroundBlurriness,n.object.backgroundIntensity=this.backgroundIntensity,n.object.backgroundRotation=this.backgroundRotation.toArray(),n.object.environmentIntensity=this.environmentIntensity,n.object.environmentRotation=this.environmentRotation.toArray(),n}},Fi=new G,Ls=new G,Fg=new G,Os=new G,Uo=new G,Io=new G,QA=new G,zg=new G,Hg=new G,Gg=new G,Vg=new Ot,kg=new Ot,Wg=new Ot,Cr=class t{constructor(e=new G,n=new G,i=new G){this.a=e,this.b=n,this.c=i}static getNormal(e,n,i,s){s.subVectors(i,n),Fi.subVectors(e,n),s.cross(Fi);let r=s.lengthSq();return r>0?s.multiplyScalar(1/Math.sqrt(r)):s.set(0,0,0)}static getBarycoord(e,n,i,s,r){Fi.subVectors(s,n),Ls.subVectors(i,n),Fg.subVectors(e,n);let a=Fi.dot(Fi),o=Fi.dot(Ls),l=Fi.dot(Fg),c=Ls.dot(Ls),h=Ls.dot(Fg),p=a*c-o*o;if(p===0)return r.set(0,0,0),null;let u=1/p,d=(c*l-o*h)*u,v=(a*h-o*l)*u;return r.set(1-d-v,v,d)}static containsPoint(e,n,i,s){return this.getBarycoord(e,n,i,s,Os)===null?!1:Os.x>=0&&Os.y>=0&&Os.x+Os.y<=1}static getInterpolation(e,n,i,s,r,a,o,l){return this.getBarycoord(e,n,i,s,Os)===null?(l.x=0,l.y=0,"z"in l&&(l.z=0),"w"in l&&(l.w=0),null):(l.setScalar(0),l.addScaledVector(r,Os.x),l.addScaledVector(a,Os.y),l.addScaledVector(o,Os.z),l)}static getInterpolatedAttribute(e,n,i,s,r,a){return Vg.setScalar(0),kg.setScalar(0),Wg.setScalar(0),Vg.fromBufferAttribute(e,n),kg.fromBufferAttribute(e,i),Wg.fromBufferAttribute(e,s),a.setScalar(0),a.addScaledVector(Vg,r.x),a.addScaledVector(kg,r.y),a.addScaledVector(Wg,r.z),a}static isFrontFacing(e,n,i,s){return Fi.subVectors(i,n),Ls.subVectors(e,n),Fi.cross(Ls).dot(s)<0}set(e,n,i){return this.a.copy(e),this.b.copy(n),this.c.copy(i),this}setFromPointsAndIndices(e,n,i,s){return this.a.copy(e[n]),this.b.copy(e[i]),this.c.copy(e[s]),this}setFromAttributeAndIndices(e,n,i,s){return this.a.fromBufferAttribute(e,n),this.b.fromBufferAttribute(e,i),this.c.fromBufferAttribute(e,s),this}clone(){return new this.constructor().copy(this)}copy(e){return this.a.copy(e.a),this.b.copy(e.b),this.c.copy(e.c),this}getArea(){return Fi.subVectors(this.c,this.b),Ls.subVectors(this.a,this.b),Fi.cross(Ls).length()*.5}getMidpoint(e){return e.addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)}getNormal(e){return t.getNormal(this.a,this.b,this.c,e)}getPlane(e){return e.setFromCoplanarPoints(this.a,this.b,this.c)}getBarycoord(e,n){return t.getBarycoord(e,this.a,this.b,this.c,n)}getInterpolation(e,n,i,s,r){return t.getInterpolation(e,this.a,this.b,this.c,n,i,s,r)}containsPoint(e){return t.containsPoint(e,this.a,this.b,this.c)}isFrontFacing(e){return t.isFrontFacing(this.a,this.b,this.c,e)}intersectsBox(e){return e.intersectsTriangle(this)}closestPointToPoint(e,n){let i=this.a,s=this.b,r=this.c,a,o;Uo.subVectors(s,i),Io.subVectors(r,i),zg.subVectors(e,i);let l=Uo.dot(zg),c=Io.dot(zg);if(l<=0&&c<=0)return n.copy(i);Hg.subVectors(e,s);let h=Uo.dot(Hg),p=Io.dot(Hg);if(h>=0&&p<=h)return n.copy(s);let u=l*p-h*c;if(u<=0&&l>=0&&h<=0)return a=l/(l-h),n.copy(i).addScaledVector(Uo,a);Gg.subVectors(e,r);let d=Uo.dot(Gg),v=Io.dot(Gg);if(v>=0&&d<=v)return n.copy(r);let M=d*c-l*v;if(M<=0&&c>=0&&v<=0)return o=c/(c-v),n.copy(i).addScaledVector(Io,o);let m=h*v-d*p;if(m<=0&&p-h>=0&&d-v>=0)return QA.subVectors(r,s),o=(p-h)/(p-h+(d-v)),n.copy(s).addScaledVector(QA,o);let f=1/(m+M+u);return a=M*f,o=u*f,n.copy(i).addScaledVector(Uo,a).addScaledVector(Io,o)}equals(e){return e.a.equals(this.a)&&e.b.equals(this.b)&&e.c.equals(this.c)}},Ur=class{constructor(e=new G(1/0,1/0,1/0),n=new G(-1/0,-1/0,-1/0)){this.isBox3=!0,this.min=e,this.max=n}set(e,n){return this.min.copy(e),this.max.copy(n),this}setFromArray(e){this.makeEmpty();for(let n=0,i=e.length;n<i;n+=3)this.expandByPoint(zi.fromArray(e,n));return this}setFromBufferAttribute(e){this.makeEmpty();for(let n=0,i=e.count;n<i;n++)this.expandByPoint(zi.fromBufferAttribute(e,n));return this}setFromPoints(e){this.makeEmpty();for(let n=0,i=e.length;n<i;n++)this.expandByPoint(e[n]);return this}setFromCenterAndSize(e,n){let i=zi.copy(n).multiplyScalar(.5);return this.min.copy(e).sub(i),this.max.copy(e).add(i),this}setFromObject(e,n=!1){return this.makeEmpty(),this.expandByObject(e,n)}clone(){return new this.constructor().copy(this)}copy(e){return this.min.copy(e.min),this.max.copy(e.max),this}makeEmpty(){return this.min.x=this.min.y=this.min.z=1/0,this.max.x=this.max.y=this.max.z=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z}getCenter(e){return this.isEmpty()?e.set(0,0,0):e.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(e){return this.isEmpty()?e.set(0,0,0):e.subVectors(this.max,this.min)}expandByPoint(e){return this.min.min(e),this.max.max(e),this}expandByVector(e){return this.min.sub(e),this.max.add(e),this}expandByScalar(e){return this.min.addScalar(-e),this.max.addScalar(e),this}expandByObject(e,n=!1){e.updateWorldMatrix(!1,!1);let i=e.geometry;if(i!==void 0){let r=i.getAttribute("position");if(n===!0&&r!==void 0&&e.isInstancedMesh!==!0)for(let a=0,o=r.count;a<o;a++)e.isMesh===!0?e.getVertexPosition(a,zi):zi.fromBufferAttribute(r,a),zi.applyMatrix4(e.matrixWorld),this.expandByPoint(zi);else e.boundingBox!==void 0?(e.boundingBox===null&&e.computeBoundingBox(),Uf.copy(e.boundingBox)):(i.boundingBox===null&&i.computeBoundingBox(),Uf.copy(i.boundingBox)),Uf.applyMatrix4(e.matrixWorld),this.union(Uf)}let s=e.children;for(let r=0,a=s.length;r<a;r++)this.expandByObject(s[r],n);return this}containsPoint(e){return e.x>=this.min.x&&e.x<=this.max.x&&e.y>=this.min.y&&e.y<=this.max.y&&e.z>=this.min.z&&e.z<=this.max.z}containsBox(e){return this.min.x<=e.min.x&&e.max.x<=this.max.x&&this.min.y<=e.min.y&&e.max.y<=this.max.y&&this.min.z<=e.min.z&&e.max.z<=this.max.z}getParameter(e,n){return n.set((e.x-this.min.x)/(this.max.x-this.min.x),(e.y-this.min.y)/(this.max.y-this.min.y),(e.z-this.min.z)/(this.max.z-this.min.z))}intersectsBox(e){return e.max.x>=this.min.x&&e.min.x<=this.max.x&&e.max.y>=this.min.y&&e.min.y<=this.max.y&&e.max.z>=this.min.z&&e.min.z<=this.max.z}intersectsSphere(e){return this.clampPoint(e.center,zi),zi.distanceToSquared(e.center)<=e.radius*e.radius}intersectsPlane(e){let n,i;return e.normal.x>0?(n=e.normal.x*this.min.x,i=e.normal.x*this.max.x):(n=e.normal.x*this.max.x,i=e.normal.x*this.min.x),e.normal.y>0?(n+=e.normal.y*this.min.y,i+=e.normal.y*this.max.y):(n+=e.normal.y*this.max.y,i+=e.normal.y*this.min.y),e.normal.z>0?(n+=e.normal.z*this.min.z,i+=e.normal.z*this.max.z):(n+=e.normal.z*this.max.z,i+=e.normal.z*this.min.z),n<=-e.constant&&i>=-e.constant}intersectsTriangle(e){if(this.isEmpty())return!1;this.getCenter(ac),If.subVectors(this.max,ac),Bo.subVectors(e.a,ac),No.subVectors(e.b,ac),Po.subVectors(e.c,ac),Tr.subVectors(No,Bo),br.subVectors(Po,No),da.subVectors(Bo,Po);let n=[0,-Tr.z,Tr.y,0,-br.z,br.y,0,-da.z,da.y,Tr.z,0,-Tr.x,br.z,0,-br.x,da.z,0,-da.x,-Tr.y,Tr.x,0,-br.y,br.x,0,-da.y,da.x,0];return!Xg(n,Bo,No,Po,If)||(n=[1,0,0,0,1,0,0,0,1],!Xg(n,Bo,No,Po,If))?!1:(Bf.crossVectors(Tr,br),n=[Bf.x,Bf.y,Bf.z],Xg(n,Bo,No,Po,If))}clampPoint(e,n){return n.copy(e).clamp(this.min,this.max)}distanceToPoint(e){return this.clampPoint(e,zi).distanceTo(e)}getBoundingSphere(e){return this.isEmpty()?e.makeEmpty():(this.getCenter(e.center),e.radius=this.getSize(zi).length()*.5),e}intersect(e){return this.min.max(e.min),this.max.min(e.max),this.isEmpty()&&this.makeEmpty(),this}union(e){return this.min.min(e.min),this.max.max(e.max),this}applyMatrix4(e){return this.isEmpty()?this:(Fs[0].set(this.min.x,this.min.y,this.min.z).applyMatrix4(e),Fs[1].set(this.min.x,this.min.y,this.max.z).applyMatrix4(e),Fs[2].set(this.min.x,this.max.y,this.min.z).applyMatrix4(e),Fs[3].set(this.min.x,this.max.y,this.max.z).applyMatrix4(e),Fs[4].set(this.max.x,this.min.y,this.min.z).applyMatrix4(e),Fs[5].set(this.max.x,this.min.y,this.max.z).applyMatrix4(e),Fs[6].set(this.max.x,this.max.y,this.min.z).applyMatrix4(e),Fs[7].set(this.max.x,this.max.y,this.max.z).applyMatrix4(e),this.setFromPoints(Fs),this)}translate(e){return this.min.add(e),this.max.add(e),this}equals(e){return e.min.equals(this.min)&&e.max.equals(this.max)}toJSON(){return{min:this.min.toArray(),max:this.max.toArray()}}fromJSON(e){return this.min.fromArray(e.min),this.max.fromArray(e.max),this}},Fs=[new G,new G,new G,new G,new G,new G,new G,new G],zi=new G,Uf=new Ur,Bo=new G,No=new G,Po=new G,Tr=new G,br=new G,da=new G,ac=new G,If=new G,Bf=new G,pa=new G;function Xg(t,e,n,i,s){for(let r=0,a=t.length-3;r<=a;r+=3){pa.fromArray(t,r);let o=s.x*Math.abs(pa.x)+s.y*Math.abs(pa.y)+s.z*Math.abs(pa.z),l=e.dot(pa),c=n.dot(pa),h=i.dot(pa);if(Math.max(-Math.max(l,c,h),Math.min(l,c,h))>o)return!1}return!0}var Jt=new G,Nf=new Ne,ww=0,Fn=class extends Zn{constructor(e,n,i=!1){if(super(),Array.isArray(e))throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");this.isBufferAttribute=!0,Object.defineProperty(this,"id",{value:ww++}),this.name="",this.array=e,this.itemSize=n,this.count=e!==void 0?e.length/n:0,this.normalized=i,this.usage=N1,this.updateRanges=[],this.gpuType=fi,this.version=0}onUploadCallback(){}set needsUpdate(e){e===!0&&this.version++}setUsage(e){return this.usage=e,this}addUpdateRange(e,n){this.updateRanges.push({start:e,count:n})}clearUpdateRanges(){this.updateRanges.length=0}copy(e){return this.name=e.name,this.array=new e.array.constructor(e.array),this.itemSize=e.itemSize,this.count=e.count,this.normalized=e.normalized,this.usage=e.usage,this.gpuType=e.gpuType,this}copyAt(e,n,i){e*=this.itemSize,i*=n.itemSize;for(let s=0,r=this.itemSize;s<r;s++)this.array[e+s]=n.array[i+s];return this}copyArray(e){return this.array.set(e),this}applyMatrix3(e){if(this.itemSize===2)for(let n=0,i=this.count;n<i;n++)Nf.fromBufferAttribute(this,n),Nf.applyMatrix3(e),this.setXY(n,Nf.x,Nf.y);else if(this.itemSize===3)for(let n=0,i=this.count;n<i;n++)Jt.fromBufferAttribute(this,n),Jt.applyMatrix3(e),this.setXYZ(n,Jt.x,Jt.y,Jt.z);return this}applyMatrix4(e){for(let n=0,i=this.count;n<i;n++)Jt.fromBufferAttribute(this,n),Jt.applyMatrix4(e),this.setXYZ(n,Jt.x,Jt.y,Jt.z);return this}applyNormalMatrix(e){for(let n=0,i=this.count;n<i;n++)Jt.fromBufferAttribute(this,n),Jt.applyNormalMatrix(e),this.setXYZ(n,Jt.x,Jt.y,Jt.z);return this}transformDirection(e){for(let n=0,i=this.count;n<i;n++)Jt.fromBufferAttribute(this,n),Jt.transformDirection(e),this.setXYZ(n,Jt.x,Jt.y,Jt.z);return this}set(e,n=0){return this.array.set(e,n),this}getComponent(e,n){let i=this.array[e*this.itemSize+n];return this.normalized&&(i=sc(i,this.array)),i}setComponent(e,n,i){return this.normalized&&(i=Qn(i,this.array)),this.array[e*this.itemSize+n]=i,this}getX(e){let n=this.array[e*this.itemSize];return this.normalized&&(n=sc(n,this.array)),n}setX(e,n){return this.normalized&&(n=Qn(n,this.array)),this.array[e*this.itemSize]=n,this}getY(e){let n=this.array[e*this.itemSize+1];return this.normalized&&(n=sc(n,this.array)),n}setY(e,n){return this.normalized&&(n=Qn(n,this.array)),this.array[e*this.itemSize+1]=n,this}getZ(e){let n=this.array[e*this.itemSize+2];return this.normalized&&(n=sc(n,this.array)),n}setZ(e,n){return this.normalized&&(n=Qn(n,this.array)),this.array[e*this.itemSize+2]=n,this}getW(e){let n=this.array[e*this.itemSize+3];return this.normalized&&(n=sc(n,this.array)),n}setW(e,n){return this.normalized&&(n=Qn(n,this.array)),this.array[e*this.itemSize+3]=n,this}setXY(e,n,i){return e*=this.itemSize,this.normalized&&(n=Qn(n,this.array),i=Qn(i,this.array)),this.array[e+0]=n,this.array[e+1]=i,this}setXYZ(e,n,i,s){return e*=this.itemSize,this.normalized&&(n=Qn(n,this.array),i=Qn(i,this.array),s=Qn(s,this.array)),this.array[e+0]=n,this.array[e+1]=i,this.array[e+2]=s,this}setXYZW(e,n,i,s,r){return e*=this.itemSize,this.normalized&&(n=Qn(n,this.array),i=Qn(i,this.array),s=Qn(s,this.array),r=Qn(r,this.array)),this.array[e+0]=n,this.array[e+1]=i,this.array[e+2]=s,this.array[e+3]=r,this}onUpload(e){return this.onUploadCallback=e,this}clone(){return new this.constructor(this.array,this.itemSize).copy(this)}toJSON(){let e={itemSize:this.itemSize,type:this.array.constructor.name,array:Array.from(this.array),normalized:this.normalized};return e.name=this.name,e.usage=this.usage,e.gpuType=this.gpuType,e}dispose(){this.dispatchEvent({type:"dispose"})}};var mc=class extends Fn{constructor(e,n,i){super(new Uint16Array(e),n,i)}};var gc=class extends Fn{constructor(e,n,i){super(new Uint32Array(e),n,i)}};var Ti=class extends Fn{constructor(e,n,i){super(new Float32Array(e),n,i)}},Cw=new Ur,oc=new G,Yg=new G,Yo=class{constructor(e=new G,n=-1){this.isSphere=!0,this.center=e,this.radius=n}set(e,n){return this.center.copy(e),this.radius=n,this}setFromPoints(e,n){let i=this.center;n!==void 0?i.copy(n):Cw.setFromPoints(e).getCenter(i);let s=0;for(let r=0,a=e.length;r<a;r++)s=Math.max(s,i.distanceToSquared(e[r]));return this.radius=Math.sqrt(s),this}copy(e){return this.center.copy(e.center),this.radius=e.radius,this}isEmpty(){return this.radius<0}makeEmpty(){return this.center.set(0,0,0),this.radius=-1,this}containsPoint(e){return e.distanceToSquared(this.center)<=this.radius*this.radius}distanceToPoint(e){return e.distanceTo(this.center)-this.radius}intersectsSphere(e){let n=this.radius+e.radius;return e.center.distanceToSquared(this.center)<=n*n}intersectsBox(e){return e.intersectsSphere(this)}intersectsPlane(e){return Math.abs(e.distanceToPoint(this.center))<=this.radius}clampPoint(e,n){let i=this.center.distanceToSquared(e);return n.copy(e),i>this.radius*this.radius&&(n.sub(this.center).normalize(),n.multiplyScalar(this.radius).add(this.center)),n}getBoundingBox(e){return this.isEmpty()?(e.makeEmpty(),e):(e.set(this.center,this.center),e.expandByScalar(this.radius),e)}applyMatrix4(e){return this.center.applyMatrix4(e),this.radius=this.radius*e.getMaxScaleOnAxis(),this}translate(e){return this.center.add(e),this}expandByPoint(e){if(this.isEmpty())return this.center.copy(e),this.radius=0,this;oc.subVectors(e,this.center);let n=oc.lengthSq();if(n>this.radius*this.radius){let i=Math.sqrt(n),s=(i-this.radius)*.5;this.center.addScaledVector(oc,s/i),this.radius+=s}return this}union(e){return e.isEmpty()?this:this.isEmpty()?(this.copy(e),this):(this.center.equals(e.center)===!0?this.radius=Math.max(this.radius,e.radius):(Yg.subVectors(e.center,this.center).setLength(e.radius),this.expandByPoint(oc.copy(e.center).add(Yg)),this.expandByPoint(oc.copy(e.center).sub(Yg))),this)}equals(e){return e.center.equals(this.center)&&e.radius===this.radius}clone(){return new this.constructor().copy(this)}toJSON(){return{radius:this.radius,center:this.center.toArray()}}fromJSON(e){return this.radius=e.radius,this.center.fromArray(e.center),this}},Rw=0,Ei=new kt,qg=new bi,Lo=new G,ci=new Ur,lc=new Ur,hn=new G,wi=class t extends Zn{constructor(){super(),this.isBufferGeometry=!0,Object.defineProperty(this,"id",{value:Rw++}),this.uuid=Nc(),this.name="",this.type="BufferGeometry",this.index=null,this.indirect=null,this.indirectOffset=0,this.attributes={},this.morphAttributes={},this.morphTargetsRelative=!1,this.groups=[],this.boundingBox=null,this.boundingSphere=null,this.drawRange={start:0,count:1/0},this.userData={},this._transformed=!1}getIndex(){return this.index}setIndex(e){return Array.isArray(e)?this.index=new(pw(e)?gc:mc)(e,1):this.index=e,this}setIndirect(e,n=0){return this.indirect=e,this.indirectOffset=n,this}getIndirect(){return this.indirect}getAttribute(e){return this.attributes[e]}setAttribute(e,n){return this.attributes[e]=n,this}deleteAttribute(e){return delete this.attributes[e],this}hasAttribute(e){return this.attributes[e]!==void 0}addGroup(e,n,i=0){this.groups.push({start:e,count:n,materialIndex:i})}clearGroups(){this.groups=[]}setDrawRange(e,n){this.drawRange.start=e,this.drawRange.count=n}applyMatrix4(e){let n=this.attributes.position;n!==void 0&&(n.applyMatrix4(e),n.needsUpdate=!0);let i=this.attributes.normal;if(i!==void 0){let r=new Pe().getNormalMatrix(e);i.applyNormalMatrix(r),i.needsUpdate=!0}let s=this.attributes.tangent;return s!==void 0&&(s.transformDirection(e),s.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this._transformed=!0,this}applyQuaternion(e){return Ei.makeRotationFromQuaternion(e),this.applyMatrix4(Ei),this}rotateX(e){return Ei.makeRotationX(e),this.applyMatrix4(Ei),this}rotateY(e){return Ei.makeRotationY(e),this.applyMatrix4(Ei),this}rotateZ(e){return Ei.makeRotationZ(e),this.applyMatrix4(Ei),this}translate(e,n,i){return Ei.makeTranslation(e,n,i),this.applyMatrix4(Ei),this}scale(e,n,i){return Ei.makeScale(e,n,i),this.applyMatrix4(Ei),this}lookAt(e){return qg.lookAt(e),qg.updateMatrix(),this.applyMatrix4(qg.matrix),this}center(){return this.computeBoundingBox(),this.boundingBox.getCenter(Lo).negate(),this.translate(Lo.x,Lo.y,Lo.z),this}setFromPoints(e){let n=this.getAttribute("position");if(n===void 0){let i=[];for(let s=0,r=e.length;s<r;s++){let a=e[s];i.push(a.x,a.y,a.z||0)}this.setAttribute("position",new Ti(i,3))}else{let i=Math.min(e.length,n.count);for(let s=0;s<i;s++){let r=e[s];n.setXYZ(s,r.x,r.y,r.z||0)}e.length>n.count&&De("BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry."),n.needsUpdate=!0}return this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new Ur);let e=this.attributes.position,n=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){Ie("BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.",this),this.boundingBox.set(new G(-1/0,-1/0,-1/0),new G(1/0,1/0,1/0));return}if(e!==void 0){if(this.boundingBox.setFromBufferAttribute(e),n)for(let i=0,s=n.length;i<s;i++){let r=n[i];ci.setFromBufferAttribute(r),this.morphTargetsRelative?(hn.addVectors(this.boundingBox.min,ci.min),this.boundingBox.expandByPoint(hn),hn.addVectors(this.boundingBox.max,ci.max),this.boundingBox.expandByPoint(hn)):(this.boundingBox.expandByPoint(ci.min),this.boundingBox.expandByPoint(ci.max))}}else this.boundingBox.makeEmpty();(isNaN(this.boundingBox.min.x)||isNaN(this.boundingBox.min.y)||isNaN(this.boundingBox.min.z))&&Ie('BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.',this)}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new Yo);let e=this.attributes.position,n=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){Ie("BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.",this),this.boundingSphere.set(new G,1/0);return}if(e){let i=this.boundingSphere.center;if(ci.setFromBufferAttribute(e),n)for(let r=0,a=n.length;r<a;r++){let o=n[r];lc.setFromBufferAttribute(o),this.morphTargetsRelative?(hn.addVectors(ci.min,lc.min),ci.expandByPoint(hn),hn.addVectors(ci.max,lc.max),ci.expandByPoint(hn)):(ci.expandByPoint(lc.min),ci.expandByPoint(lc.max))}ci.getCenter(i);let s=0;for(let r=0,a=e.count;r<a;r++)hn.fromBufferAttribute(e,r),s=Math.max(s,i.distanceToSquared(hn));if(n)for(let r=0,a=n.length;r<a;r++){let o=n[r],l=this.morphTargetsRelative;for(let c=0,h=o.count;c<h;c++)hn.fromBufferAttribute(o,c),l&&(Lo.fromBufferAttribute(e,c),hn.add(Lo)),s=Math.max(s,i.distanceToSquared(hn))}this.boundingSphere.radius=Math.sqrt(s),isNaN(this.boundingSphere.radius)&&Ie('BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.',this)}}computeTangents(){let e=this.index,n=this.attributes;if(e===null||n.position===void 0||n.normal===void 0||n.uv===void 0){Ie("BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");return}let i=n.position,s=n.normal,r=n.uv,a=this.getAttribute("tangent");(a===void 0||a.count!==i.count)&&(a=new Fn(new Float32Array(4*i.count),4),this.setAttribute("tangent",a));let o=[],l=[];for(let y=0;y<i.count;y++)o[y]=new G,l[y]=new G;let c=new G,h=new G,p=new G,u=new Ne,d=new Ne,v=new Ne,M=new G,m=new G;function f(y,b,R){c.fromBufferAttribute(i,y),h.fromBufferAttribute(i,b),p.fromBufferAttribute(i,R),u.fromBufferAttribute(r,y),d.fromBufferAttribute(r,b),v.fromBufferAttribute(r,R),h.sub(c),p.sub(c),d.sub(u),v.sub(u);let N=1/(d.x*v.y-v.x*d.y);isFinite(N)&&(M.copy(h).multiplyScalar(v.y).addScaledVector(p,-d.y).multiplyScalar(N),m.copy(p).multiplyScalar(d.x).addScaledVector(h,-v.x).multiplyScalar(N),o[y].add(M),o[b].add(M),o[R].add(M),l[y].add(m),l[b].add(m),l[R].add(m))}let g=this.groups;g.length===0&&(g=[{start:0,count:e.count}]);for(let y=0,b=g.length;y<b;++y){let R=g[y],N=R.start,F=R.count;for(let k=N,B=N+F;k<B;k+=3)f(e.getX(k+0),e.getX(k+1),e.getX(k+2))}let S=new G,_=new G,E=new G,T=new G;function C(y){E.fromBufferAttribute(s,y),T.copy(E);let b=o[y];S.copy(b),S.sub(E.multiplyScalar(E.dot(b))).normalize(),_.crossVectors(T,b);let N=_.dot(l[y])<0?-1:1;a.setXYZW(y,S.x,S.y,S.z,N)}for(let y=0,b=g.length;y<b;++y){let R=g[y],N=R.start,F=R.count;for(let k=N,B=N+F;k<B;k+=3)C(e.getX(k+0)),C(e.getX(k+1)),C(e.getX(k+2))}this._transformed=!0}computeVertexNormals(){let e=this.index,n=this.getAttribute("position");if(n!==void 0){let i=this.getAttribute("normal");if(i===void 0||i.count!==n.count)i=new Fn(new Float32Array(n.count*3),3),this.setAttribute("normal",i);else for(let u=0,d=i.count;u<d;u++)i.setXYZ(u,0,0,0);let s=new G,r=new G,a=new G,o=new G,l=new G,c=new G,h=new G,p=new G;if(e)for(let u=0,d=e.count;u<d;u+=3){let v=e.getX(u+0),M=e.getX(u+1),m=e.getX(u+2);s.fromBufferAttribute(n,v),r.fromBufferAttribute(n,M),a.fromBufferAttribute(n,m),h.subVectors(a,r),p.subVectors(s,r),h.cross(p),o.fromBufferAttribute(i,v),l.fromBufferAttribute(i,M),c.fromBufferAttribute(i,m),o.add(h),l.add(h),c.add(h),i.setXYZ(v,o.x,o.y,o.z),i.setXYZ(M,l.x,l.y,l.z),i.setXYZ(m,c.x,c.y,c.z)}else for(let u=0,d=n.count;u<d;u+=3)s.fromBufferAttribute(n,u+0),r.fromBufferAttribute(n,u+1),a.fromBufferAttribute(n,u+2),h.subVectors(a,r),p.subVectors(s,r),h.cross(p),i.setXYZ(u+0,h.x,h.y,h.z),i.setXYZ(u+1,h.x,h.y,h.z),i.setXYZ(u+2,h.x,h.y,h.z);this.normalizeNormals(),i.needsUpdate=!0}}normalizeNormals(){let e=this.attributes.normal;for(let n=0,i=e.count;n<i;n++)hn.fromBufferAttribute(e,n),hn.normalize(),e.setXYZ(n,hn.x,hn.y,hn.z)}toNonIndexed(){function e(o,l){let c=o.array,h=o.itemSize,p=o.normalized,u=new c.constructor(l.length*h),d=0,v=0;for(let M=0,m=l.length;M<m;M++){o.isInterleavedBufferAttribute?d=l[M]*o.data.stride+o.offset:d=l[M]*h;for(let f=0;f<h;f++)u[v++]=c[d++]}return new Fn(u,h,p)}if(this.index===null)return De("BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."),this;let n=new t,i=this.index.array,s=this.attributes;for(let o in s){let l=s[o],c=e(l,i);n.setAttribute(o,c)}let r=this.morphAttributes;for(let o in r){let l=[],c=r[o];for(let h=0,p=c.length;h<p;h++){let u=c[h],d=e(u,i);l.push(d)}n.morphAttributes[o]=l}n.morphTargetsRelative=this.morphTargetsRelative;let a=this.groups;for(let o=0,l=a.length;o<l;o++){let c=a[o];n.addGroup(c.start,c.count,c.materialIndex)}return n}toJSON(){let e={metadata:{version:4.7,type:"BufferGeometry",generator:"BufferGeometry.toJSON"}};if(e.uuid=this.uuid,e.type=this.parameters!==void 0&&this._transformed===!0?"BufferGeometry":this.type,e.name=this.name,Object.keys(this.userData).length>0&&(e.userData=this.userData),this.parameters!==void 0&&this._transformed!==!0){let l=this.parameters;for(let c in l)l[c]!==void 0&&(e[c]=l[c]);return e}e.data={attributes:{}};let n=this.index;n!==null&&(e.data.index={type:n.array.constructor.name,array:Array.prototype.slice.call(n.array)});let i=this.attributes;for(let l in i){let c=i[l];e.data.attributes[l]=c.toJSON(e.data)}let s={},r=!1;for(let l in this.morphAttributes){let c=this.morphAttributes[l],h=[];for(let p=0,u=c.length;p<u;p++){let d=c[p];h.push(d.toJSON(e.data))}h.length>0&&(s[l]=h,r=!0)}r&&(e.data.morphAttributes=s,e.data.morphTargetsRelative=this.morphTargetsRelative);let a=this.groups;a.length>0&&(e.data.groups=JSON.parse(JSON.stringify(a)));let o=this.boundingSphere;return o!==null&&(e.data.boundingSphere=o.toJSON()),e}clone(){return new this.constructor().copy(this)}copy(e){this.index=null,this.attributes={},this.morphAttributes={},this.groups=[],this.boundingBox=null,this.boundingSphere=null;let n={};this.name=e.name;let i=e.index;i!==null&&this.setIndex(i.clone());let s=e.attributes;for(let c in s){let h=s[c];this.setAttribute(c,h.clone(n))}let r=e.morphAttributes;for(let c in r){let h=[],p=r[c];for(let u=0,d=p.length;u<d;u++)h.push(p[u].clone(n));this.morphAttributes[c]=h}this.morphTargetsRelative=e.morphTargetsRelative;let a=e.groups;for(let c=0,h=a.length;c<h;c++){let p=a[c];this.addGroup(p.start,p.count,p.materialIndex)}let o=e.boundingBox;o!==null&&(this.boundingBox=o.clone());let l=e.boundingSphere;return l!==null&&(this.boundingSphere=l.clone()),this.drawRange.start=e.drawRange.start,this.drawRange.count=e.drawRange.count,this.userData=e.userData,this._transformed=e._transformed,this}dispose(){this.dispatchEvent({type:"dispose"})}};var Qg=new G,Dw=new G,Uw=new Pe,Hi=class{constructor(e=new G(1,0,0),n=0){this.isPlane=!0,this.normal=e,this.constant=n}set(e,n){return this.normal.copy(e),this.constant=n,this}setComponents(e,n,i,s){return this.normal.set(e,n,i),this.constant=s,this}setFromNormalAndCoplanarPoint(e,n){return this.normal.copy(e),this.constant=-n.dot(this.normal),this}setFromCoplanarPoints(e,n,i){let s=Qg.subVectors(i,n).cross(Dw.subVectors(e,n)).normalize();return this.setFromNormalAndCoplanarPoint(s,e),this}copy(e){return this.normal.copy(e.normal),this.constant=e.constant,this}normalize(){let e=1/this.normal.length();return this.normal.multiplyScalar(e),this.constant*=e,this}negate(){return this.constant*=-1,this.normal.negate(),this}distanceToPoint(e){return this.normal.dot(e)+this.constant}distanceToSphere(e){return this.distanceToPoint(e.center)-e.radius}projectPoint(e,n){return n.copy(e).addScaledVector(this.normal,-this.distanceToPoint(e))}intersectLine(e,n,i=!0){let s=e.delta(Qg),r=this.normal.dot(s);if(r===0)return this.distanceToPoint(e.start)===0?n.copy(e.start):null;let a=-(e.start.dot(this.normal)+this.constant)/r;return i===!0&&(a<0||a>1)?null:n.copy(e.start).addScaledVector(s,a)}intersectsLine(e){let n=this.distanceToPoint(e.start),i=this.distanceToPoint(e.end);return n<0&&i>0||i<0&&n>0}intersectsBox(e){return e.intersectsPlane(this)}intersectsSphere(e){return e.intersectsPlane(this)}coplanarPoint(e){return e.copy(this.normal).multiplyScalar(-this.constant)}applyMatrix4(e,n){let i=n||Uw.getNormalMatrix(e),s=this.coplanarPoint(Qg).applyMatrix4(e),r=this.normal.applyMatrix3(i).normalize();return this.constant=-s.dot(r),this}translate(e){return this.constant-=e.dot(this.normal),this}equals(e){return e.normal.equals(this.normal)&&e.constant===this.constant}clone(){return new this.constructor().copy(this)}toJSON(){return{normal:this.normal.toArray(),constant:this.constant}}fromJSON(e){return this.normal.fromArray(e.normal),this.constant=e.constant,this}},Iw=0,Vi=class extends Zn{constructor(){super(),this.isMaterial=!0,Object.defineProperty(this,"id",{value:Iw++}),this.uuid=Nc(),this.name="",this.type="Material",this.blending=Zo,this.side=ps,this.vertexColors=!1,this.opacity=1,this.transparent=!1,this.alphaHash=!1,this.blendSrc=l0,this.blendDst=c0,this.blendEquation=_a,this.blendSrcAlpha=null,this.blendDstAlpha=null,this.blendEquationAlpha=null,this.blendColor=new Ye(0,0,0),this.blendAlpha=0,this.depthFunc=Vo,this.depthTest=!0,this.depthWrite=!0,this.stencilWriteMask=255,this.stencilFunc=w1,this.stencilRef=0,this.stencilFuncMask=255,this.stencilFail=Qf,this.stencilZFail=Qf,this.stencilZPass=Qf,this.stencilWrite=!1,this.clippingPlanes=null,this.clipIntersection=!1,this.clipShadows=!1,this.shadowSide=null,this.colorWrite=!0,this.precision=null,this.polygonOffset=!1,this.polygonOffsetFactor=0,this.polygonOffsetUnits=0,this.dithering=!1,this.alphaToCoverage=!1,this.premultipliedAlpha=!1,this.forceSinglePass=!1,this.allowOverride=!0,this.visible=!0,this.toneMapped=!0,this.userData={},this.version=0,this._alphaTest=0}get alphaTest(){return this._alphaTest}set alphaTest(e){this._alphaTest>0!=e>0&&this.version++,this._alphaTest=e}onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){return this.onBeforeCompile.toString()}setValues(e){if(e!==void 0)for(let n in e){let i=e[n];if(i===void 0){De(`Material: parameter '${n}' has value of undefined.`);continue}let s=this[n];if(s===void 0){De(`Material: '${n}' is not a property of THREE.${this.type}.`);continue}s&&s.isColor?s.set(i):s&&s.isVector2&&i&&i.isVector2||s&&s.isEuler&&i&&i.isEuler||s&&s.isVector3&&i&&i.isVector3?s.copy(i):this[n]=i}}toJSON(e){let n=e===void 0||typeof e=="string";n&&(e={textures:{},images:{}});let i={metadata:{version:4.7,type:"Material",generator:"Material.toJSON"}};i.uuid=this.uuid,i.type=this.type,i.blending=this.blending,i.side=this.side,i.shadowSide=this.shadowSide,i.vertexColors=this.vertexColors,i.opacity=this.opacity,i.transparent=this.transparent,i.blendSrc=this.blendSrc,i.blendDst=this.blendDst,i.blendEquation=this.blendEquation,i.blendSrcAlpha=this.blendSrcAlpha,i.blendDstAlpha=this.blendDstAlpha,i.blendEquationAlpha=this.blendEquationAlpha,i.blendColor=this.blendColor.getHex(),i.blendAlpha=this.blendAlpha,i.depthFunc=this.depthFunc,i.depthTest=this.depthTest,i.depthWrite=this.depthWrite,i.colorWrite=this.colorWrite,i.clipIntersection=this.clipIntersection,i.clipShadows=this.clipShadows,i.stencilWriteMask=this.stencilWriteMask,i.stencilFunc=this.stencilFunc,i.stencilRef=this.stencilRef,i.stencilFuncMask=this.stencilFuncMask,i.stencilFail=this.stencilFail,i.stencilZFail=this.stencilZFail,i.stencilZPass=this.stencilZPass,i.stencilWrite=this.stencilWrite,i.polygonOffset=this.polygonOffset,i.polygonOffsetFactor=this.polygonOffsetFactor,i.polygonOffsetUnits=this.polygonOffsetUnits,i.dithering=this.dithering,i.alphaTest=this.alphaTest,i.alphaHash=this.alphaHash,i.alphaToCoverage=this.alphaToCoverage,i.premultipliedAlpha=this.premultipliedAlpha,i.forceSinglePass=this.forceSinglePass,i.allowOverride=this.allowOverride,i.visible=this.visible,i.toneMapped=this.toneMapped,i.name=this.name,this.color&&this.color.isColor&&(i.color=this.color.getHex()),this.roughness!==void 0&&(i.roughness=this.roughness),this.metalness!==void 0&&(i.metalness=this.metalness),this.sheen!==void 0&&(i.sheen=this.sheen),this.sheenColor&&this.sheenColor.isColor&&(i.sheenColor=this.sheenColor.getHex()),this.sheenRoughness!==void 0&&(i.sheenRoughness=this.sheenRoughness),this.emissive&&this.emissive.isColor&&(i.emissive=this.emissive.getHex()),this.emissiveIntensity!==void 0&&(i.emissiveIntensity=this.emissiveIntensity),this.specular&&this.specular.isColor&&(i.specular=this.specular.getHex()),this.specularIntensity!==void 0&&(i.specularIntensity=this.specularIntensity),this.specularColor&&this.specularColor.isColor&&(i.specularColor=this.specularColor.getHex()),this.shininess!==void 0&&(i.shininess=this.shininess),this.clearcoat!==void 0&&(i.clearcoat=this.clearcoat),this.clearcoatRoughness!==void 0&&(i.clearcoatRoughness=this.clearcoatRoughness),this.clearcoatMap&&this.clearcoatMap.isTexture&&(i.clearcoatMap=this.clearcoatMap.toJSON(e).uuid),this.clearcoatRoughnessMap&&this.clearcoatRoughnessMap.isTexture&&(i.clearcoatRoughnessMap=this.clearcoatRoughnessMap.toJSON(e).uuid),this.clearcoatNormalMap&&this.clearcoatNormalMap.isTexture&&(i.clearcoatNormalMap=this.clearcoatNormalMap.toJSON(e).uuid,i.clearcoatNormalScale=this.clearcoatNormalScale.toArray()),this.sheenColorMap&&this.sheenColorMap.isTexture&&(i.sheenColorMap=this.sheenColorMap.toJSON(e).uuid),this.sheenRoughnessMap&&this.sheenRoughnessMap.isTexture&&(i.sheenRoughnessMap=this.sheenRoughnessMap.toJSON(e).uuid),this.dispersion!==void 0&&(i.dispersion=this.dispersion),this.retroreflectivity!==void 0&&(i.retroreflectivity=this.retroreflectivity),this.iridescence!==void 0&&(i.iridescence=this.iridescence),this.iridescenceIOR!==void 0&&(i.iridescenceIOR=this.iridescenceIOR),this.iridescenceThicknessRange!==void 0&&(i.iridescenceThicknessRange=this.iridescenceThicknessRange),this.iridescenceMap&&this.iridescenceMap.isTexture&&(i.iridescenceMap=this.iridescenceMap.toJSON(e).uuid),this.iridescenceThicknessMap&&this.iridescenceThicknessMap.isTexture&&(i.iridescenceThicknessMap=this.iridescenceThicknessMap.toJSON(e).uuid),this.anisotropy!==void 0&&(i.anisotropy=this.anisotropy),this.anisotropyRotation!==void 0&&(i.anisotropyRotation=this.anisotropyRotation),this.anisotropyMap&&this.anisotropyMap.isTexture&&(i.anisotropyMap=this.anisotropyMap.toJSON(e).uuid),this.map&&this.map.isTexture&&(i.map=this.map.toJSON(e).uuid),this.matcap&&this.matcap.isTexture&&(i.matcap=this.matcap.toJSON(e).uuid),this.alphaMap&&this.alphaMap.isTexture&&(i.alphaMap=this.alphaMap.toJSON(e).uuid),this.lightMap&&this.lightMap.isTexture&&(i.lightMap=this.lightMap.toJSON(e).uuid,i.lightMapIntensity=this.lightMapIntensity),this.aoMap&&this.aoMap.isTexture&&(i.aoMap=this.aoMap.toJSON(e).uuid,i.aoMapIntensity=this.aoMapIntensity),this.bumpMap&&this.bumpMap.isTexture&&(i.bumpMap=this.bumpMap.toJSON(e).uuid,i.bumpScale=this.bumpScale),this.normalMap&&this.normalMap.isTexture&&(i.normalMap=this.normalMap.toJSON(e).uuid,i.normalMapType=this.normalMapType,i.normalScale=this.normalScale.toArray()),this.displacementMap&&this.displacementMap.isTexture&&(i.displacementMap=this.displacementMap.toJSON(e).uuid,i.displacementScale=this.displacementScale,i.displacementBias=this.displacementBias),this.roughnessMap&&this.roughnessMap.isTexture&&(i.roughnessMap=this.roughnessMap.toJSON(e).uuid),this.metalnessMap&&this.metalnessMap.isTexture&&(i.metalnessMap=this.metalnessMap.toJSON(e).uuid),this.emissiveMap&&this.emissiveMap.isTexture&&(i.emissiveMap=this.emissiveMap.toJSON(e).uuid),this.specularMap&&this.specularMap.isTexture&&(i.specularMap=this.specularMap.toJSON(e).uuid),this.specularIntensityMap&&this.specularIntensityMap.isTexture&&(i.specularIntensityMap=this.specularIntensityMap.toJSON(e).uuid),this.specularColorMap&&this.specularColorMap.isTexture&&(i.specularColorMap=this.specularColorMap.toJSON(e).uuid),this.envMap&&this.envMap.isTexture&&(i.envMap=this.envMap.toJSON(e).uuid,this.combine!==void 0&&(i.combine=this.combine)),this.envMapRotation!==void 0&&(i.envMapRotation=this.envMapRotation.toArray()),this.envMapIntensity!==void 0&&(i.envMapIntensity=this.envMapIntensity),this.reflectivity!==void 0&&(i.reflectivity=this.reflectivity),this.refractionRatio!==void 0&&(i.refractionRatio=this.refractionRatio),this.gradientMap&&this.gradientMap.isTexture&&(i.gradientMap=this.gradientMap.toJSON(e).uuid),this.transmission!==void 0&&(i.transmission=this.transmission),this.transmissionMap&&this.transmissionMap.isTexture&&(i.transmissionMap=this.transmissionMap.toJSON(e).uuid),this.thickness!==void 0&&(i.thickness=this.thickness),this.thicknessMap&&this.thicknessMap.isTexture&&(i.thicknessMap=this.thicknessMap.toJSON(e).uuid),this.attenuationDistance!==void 0&&(i.attenuationDistance=this.attenuationDistance),this.attenuationColor!==void 0&&(i.attenuationColor=this.attenuationColor.getHex()),this.size!==void 0&&(i.size=this.size),this.sizeAttenuation!==void 0&&(i.sizeAttenuation=this.sizeAttenuation),Array.isArray(this.clippingPlanes)&&this.clippingPlanes.length>0&&(i.clippingPlanes=this.clippingPlanes.map(r=>r.toJSON())),this.rotation!==void 0&&(i.rotation=this.rotation),this.depthPacking!==void 0&&(i.depthPacking=this.depthPacking),this.linewidth!==void 0&&(i.linewidth=this.linewidth),this.linecap!==void 0&&(i.linecap=this.linecap),this.linejoin!==void 0&&(i.linejoin=this.linejoin),this.dashSize!==void 0&&(i.dashSize=this.dashSize),this.gapSize!==void 0&&(i.gapSize=this.gapSize),this.scale!==void 0&&(i.scale=this.scale),this.wireframe!==void 0&&(i.wireframe=this.wireframe),this.wireframeLinewidth!==void 0&&(i.wireframeLinewidth=this.wireframeLinewidth),this.wireframeLinecap!==void 0&&(i.wireframeLinecap=this.wireframeLinecap),this.wireframeLinejoin!==void 0&&(i.wireframeLinejoin=this.wireframeLinejoin),this.flatShading!==void 0&&(i.flatShading=this.flatShading),this.fog!==void 0&&(i.fog=this.fog),Object.keys(this.userData).length>0&&(i.userData=this.userData);function s(r){let a=[];for(let o in r){let l=r[o];delete l.metadata,a.push(l)}return a}if(n){let r=s(e.textures),a=s(e.images);r.length>0&&(i.textures=r),a.length>0&&(i.images=a)}return i}fromJSON(e,n){if(e.uuid!==void 0&&(this.uuid=e.uuid),e.name!==void 0&&(this.name=e.name),e.color!==void 0&&this.color!==void 0&&this.color.setHex(e.color),e.roughness!==void 0&&(this.roughness=e.roughness),e.metalness!==void 0&&(this.metalness=e.metalness),e.sheen!==void 0&&(this.sheen=e.sheen),e.sheenColor!==void 0&&(this.sheenColor=new Ye().setHex(e.sheenColor)),e.sheenRoughness!==void 0&&(this.sheenRoughness=e.sheenRoughness),e.emissive!==void 0&&this.emissive!==void 0&&this.emissive.setHex(e.emissive),e.specular!==void 0&&this.specular!==void 0&&this.specular.setHex(e.specular),e.specularIntensity!==void 0&&(this.specularIntensity=e.specularIntensity),e.specularColor!==void 0&&this.specularColor!==void 0&&this.specularColor.setHex(e.specularColor),e.shininess!==void 0&&(this.shininess=e.shininess),e.clearcoat!==void 0&&(this.clearcoat=e.clearcoat),e.clearcoatRoughness!==void 0&&(this.clearcoatRoughness=e.clearcoatRoughness),e.dispersion!==void 0&&(this.dispersion=e.dispersion),e.retroreflectivity!==void 0&&(this.retroreflectivity=e.retroreflectivity),e.iridescence!==void 0&&(this.iridescence=e.iridescence),e.iridescenceIOR!==void 0&&(this.iridescenceIOR=e.iridescenceIOR),e.iridescenceThicknessRange!==void 0&&(this.iridescenceThicknessRange=e.iridescenceThicknessRange),e.transmission!==void 0&&(this.transmission=e.transmission),e.thickness!==void 0&&(this.thickness=e.thickness),e.attenuationDistance!==void 0&&(this.attenuationDistance=e.attenuationDistance),e.attenuationColor!==void 0&&this.attenuationColor!==void 0&&this.attenuationColor.setHex(e.attenuationColor),e.anisotropy!==void 0&&(this.anisotropy=e.anisotropy),e.anisotropyRotation!==void 0&&(this.anisotropyRotation=e.anisotropyRotation),e.fog!==void 0&&(this.fog=e.fog),e.flatShading!==void 0&&(this.flatShading=e.flatShading),e.blending!==void 0&&(this.blending=e.blending),e.combine!==void 0&&(this.combine=e.combine),e.side!==void 0&&(this.side=e.side),e.shadowSide!==void 0&&(this.shadowSide=e.shadowSide),e.opacity!==void 0&&(this.opacity=e.opacity),e.transparent!==void 0&&(this.transparent=e.transparent),e.alphaTest!==void 0&&(this.alphaTest=e.alphaTest),e.alphaHash!==void 0&&(this.alphaHash=e.alphaHash),e.depthFunc!==void 0&&(this.depthFunc=e.depthFunc),e.depthTest!==void 0&&(this.depthTest=e.depthTest),e.depthWrite!==void 0&&(this.depthWrite=e.depthWrite),e.colorWrite!==void 0&&(this.colorWrite=e.colorWrite),e.clippingPlanes!==void 0&&(this.clippingPlanes=e.clippingPlanes.map(i=>new Hi().fromJSON(i))),e.clipIntersection!==void 0&&(this.clipIntersection=e.clipIntersection),e.clipShadows!==void 0&&(this.clipShadows=e.clipShadows),e.depthPacking!==void 0&&(this.depthPacking=e.depthPacking),e.blendSrc!==void 0&&(this.blendSrc=e.blendSrc),e.blendDst!==void 0&&(this.blendDst=e.blendDst),e.blendEquation!==void 0&&(this.blendEquation=e.blendEquation),e.blendSrcAlpha!==void 0&&(this.blendSrcAlpha=e.blendSrcAlpha),e.blendDstAlpha!==void 0&&(this.blendDstAlpha=e.blendDstAlpha),e.blendEquationAlpha!==void 0&&(this.blendEquationAlpha=e.blendEquationAlpha),e.blendColor!==void 0&&this.blendColor!==void 0&&this.blendColor.setHex(e.blendColor),e.blendAlpha!==void 0&&(this.blendAlpha=e.blendAlpha),e.stencilWriteMask!==void 0&&(this.stencilWriteMask=e.stencilWriteMask),e.stencilFunc!==void 0&&(this.stencilFunc=e.stencilFunc),e.stencilRef!==void 0&&(this.stencilRef=e.stencilRef),e.stencilFuncMask!==void 0&&(this.stencilFuncMask=e.stencilFuncMask),e.stencilFail!==void 0&&(this.stencilFail=e.stencilFail),e.stencilZFail!==void 0&&(this.stencilZFail=e.stencilZFail),e.stencilZPass!==void 0&&(this.stencilZPass=e.stencilZPass),e.stencilWrite!==void 0&&(this.stencilWrite=e.stencilWrite),e.wireframe!==void 0&&(this.wireframe=e.wireframe),e.wireframeLinewidth!==void 0&&(this.wireframeLinewidth=e.wireframeLinewidth),e.wireframeLinecap!==void 0&&(this.wireframeLinecap=e.wireframeLinecap),e.wireframeLinejoin!==void 0&&(this.wireframeLinejoin=e.wireframeLinejoin),e.rotation!==void 0&&(this.rotation=e.rotation),e.linewidth!==void 0&&(this.linewidth=e.linewidth),e.linecap!==void 0&&(this.linecap=e.linecap),e.linejoin!==void 0&&(this.linejoin=e.linejoin),e.dashSize!==void 0&&(this.dashSize=e.dashSize),e.gapSize!==void 0&&(this.gapSize=e.gapSize),e.scale!==void 0&&(this.scale=e.scale),e.polygonOffset!==void 0&&(this.polygonOffset=e.polygonOffset),e.polygonOffsetFactor!==void 0&&(this.polygonOffsetFactor=e.polygonOffsetFactor),e.polygonOffsetUnits!==void 0&&(this.polygonOffsetUnits=e.polygonOffsetUnits),e.dithering!==void 0&&(this.dithering=e.dithering),e.alphaToCoverage!==void 0&&(this.alphaToCoverage=e.alphaToCoverage),e.premultipliedAlpha!==void 0&&(this.premultipliedAlpha=e.premultipliedAlpha),e.forceSinglePass!==void 0&&(this.forceSinglePass=e.forceSinglePass),e.allowOverride!==void 0&&(this.allowOverride=e.allowOverride),e.visible!==void 0&&(this.visible=e.visible),e.toneMapped!==void 0&&(this.toneMapped=e.toneMapped),e.userData!==void 0&&(this.userData=e.userData),e.vertexColors!==void 0&&(typeof e.vertexColors=="number"?this.vertexColors=e.vertexColors>0:this.vertexColors=e.vertexColors),e.size!==void 0&&(this.size=e.size),e.sizeAttenuation!==void 0&&(this.sizeAttenuation=e.sizeAttenuation),e.map!==void 0&&(this.map=n[e.map]||null),e.matcap!==void 0&&(this.matcap=n[e.matcap]||null),e.alphaMap!==void 0&&(this.alphaMap=n[e.alphaMap]||null),e.bumpMap!==void 0&&(this.bumpMap=n[e.bumpMap]||null),e.bumpScale!==void 0&&(this.bumpScale=e.bumpScale),e.normalMap!==void 0&&(this.normalMap=n[e.normalMap]||null),e.normalMapType!==void 0&&(this.normalMapType=e.normalMapType),e.normalScale!==void 0){let i=e.normalScale;Array.isArray(i)===!1&&(i=[i,i]),this.normalScale=new Ne().fromArray(i)}return e.displacementMap!==void 0&&(this.displacementMap=n[e.displacementMap]||null),e.displacementScale!==void 0&&(this.displacementScale=e.displacementScale),e.displacementBias!==void 0&&(this.displacementBias=e.displacementBias),e.roughnessMap!==void 0&&(this.roughnessMap=n[e.roughnessMap]||null),e.metalnessMap!==void 0&&(this.metalnessMap=n[e.metalnessMap]||null),e.emissiveMap!==void 0&&(this.emissiveMap=n[e.emissiveMap]||null),e.emissiveIntensity!==void 0&&(this.emissiveIntensity=e.emissiveIntensity),e.specularMap!==void 0&&(this.specularMap=n[e.specularMap]||null),e.specularIntensityMap!==void 0&&(this.specularIntensityMap=n[e.specularIntensityMap]||null),e.specularColorMap!==void 0&&(this.specularColorMap=n[e.specularColorMap]||null),e.envMap!==void 0&&(this.envMap=n[e.envMap]||null),e.envMapRotation!==void 0&&this.envMapRotation.fromArray(e.envMapRotation),e.envMapIntensity!==void 0&&(this.envMapIntensity=e.envMapIntensity),e.reflectivity!==void 0&&(this.reflectivity=e.reflectivity),e.refractionRatio!==void 0&&(this.refractionRatio=e.refractionRatio),e.lightMap!==void 0&&(this.lightMap=n[e.lightMap]||null),e.lightMapIntensity!==void 0&&(this.lightMapIntensity=e.lightMapIntensity),e.aoMap!==void 0&&(this.aoMap=n[e.aoMap]||null),e.aoMapIntensity!==void 0&&(this.aoMapIntensity=e.aoMapIntensity),e.gradientMap!==void 0&&(this.gradientMap=n[e.gradientMap]||null),e.clearcoatMap!==void 0&&(this.clearcoatMap=n[e.clearcoatMap]||null),e.clearcoatRoughnessMap!==void 0&&(this.clearcoatRoughnessMap=n[e.clearcoatRoughnessMap]||null),e.clearcoatNormalMap!==void 0&&(this.clearcoatNormalMap=n[e.clearcoatNormalMap]||null),e.clearcoatNormalScale!==void 0&&(this.clearcoatNormalScale=new Ne().fromArray(e.clearcoatNormalScale)),e.iridescenceMap!==void 0&&(this.iridescenceMap=n[e.iridescenceMap]||null),e.iridescenceThicknessMap!==void 0&&(this.iridescenceThicknessMap=n[e.iridescenceThicknessMap]||null),e.transmissionMap!==void 0&&(this.transmissionMap=n[e.transmissionMap]||null),e.thicknessMap!==void 0&&(this.thicknessMap=n[e.thicknessMap]||null),e.anisotropyMap!==void 0&&(this.anisotropyMap=n[e.anisotropyMap]||null),e.sheenColorMap!==void 0&&(this.sheenColorMap=n[e.sheenColorMap]||null),e.sheenRoughnessMap!==void 0&&(this.sheenRoughnessMap=n[e.sheenRoughnessMap]||null),this}clone(){return new this.constructor().copy(this)}copy(e){this.name=e.name,this.blending=e.blending,this.side=e.side,this.vertexColors=e.vertexColors,this.opacity=e.opacity,this.transparent=e.transparent,this.blendSrc=e.blendSrc,this.blendDst=e.blendDst,this.blendEquation=e.blendEquation,this.blendSrcAlpha=e.blendSrcAlpha,this.blendDstAlpha=e.blendDstAlpha,this.blendEquationAlpha=e.blendEquationAlpha,this.blendColor.copy(e.blendColor),this.blendAlpha=e.blendAlpha,this.depthFunc=e.depthFunc,this.depthTest=e.depthTest,this.depthWrite=e.depthWrite,this.stencilWriteMask=e.stencilWriteMask,this.stencilFunc=e.stencilFunc,this.stencilRef=e.stencilRef,this.stencilFuncMask=e.stencilFuncMask,this.stencilFail=e.stencilFail,this.stencilZFail=e.stencilZFail,this.stencilZPass=e.stencilZPass,this.stencilWrite=e.stencilWrite;let n=e.clippingPlanes,i=null;if(n!==null){let s=n.length;i=new Array(s);for(let r=0;r!==s;++r)i[r]=n[r].clone()}return this.clippingPlanes=i,this.clipIntersection=e.clipIntersection,this.clipShadows=e.clipShadows,this.shadowSide=e.shadowSide,this.colorWrite=e.colorWrite,this.precision=e.precision,this.polygonOffset=e.polygonOffset,this.polygonOffsetFactor=e.polygonOffsetFactor,this.polygonOffsetUnits=e.polygonOffsetUnits,this.dithering=e.dithering,this.alphaTest=e.alphaTest,this.alphaHash=e.alphaHash,this.alphaToCoverage=e.alphaToCoverage,this.premultipliedAlpha=e.premultipliedAlpha,this.forceSinglePass=e.forceSinglePass,this.allowOverride=e.allowOverride,this.visible=e.visible,this.toneMapped=e.toneMapped,this.userData=JSON.parse(JSON.stringify(e.userData)),this}dispose(){this.dispatchEvent({type:"dispose"})}set needsUpdate(e){e===!0&&this.version++}};var zs=new G,Zg=new G,Pf=new G,Lf=new G,lh=class{constructor(e=new G,n=new G(0,0,-1)){this.origin=e,this.direction=n}set(e,n){return this.origin.copy(e),this.direction.copy(n),this}copy(e){return this.origin.copy(e.origin),this.direction.copy(e.direction),this}at(e,n){return n.copy(this.origin).addScaledVector(this.direction,e)}lookAt(e){return this.direction.copy(e).sub(this.origin).normalize(),this}recast(e){return this.origin.copy(this.at(e,zs)),this}closestPointToPoint(e,n){n.subVectors(e,this.origin);let i=n.dot(this.direction);return i<0?n.copy(this.origin):n.copy(this.origin).addScaledVector(this.direction,i)}distanceToPoint(e){return Math.sqrt(this.distanceSqToPoint(e))}distanceSqToPoint(e){let n=zs.subVectors(e,this.origin).dot(this.direction);return n<0?this.origin.distanceToSquared(e):(zs.copy(this.origin).addScaledVector(this.direction,n),zs.distanceToSquared(e))}distanceSqToSegment(e,n,i,s){Zg.copy(e).add(n).multiplyScalar(.5),Pf.copy(n).sub(e).normalize(),Lf.copy(this.origin).sub(Zg);let r=e.distanceTo(n)*.5,a=-this.direction.dot(Pf),o=Lf.dot(this.direction),l=-Lf.dot(Pf),c=Lf.lengthSq(),h=Math.abs(1-a*a),p,u,d,v;if(h>0)if(p=a*l-o,u=a*o-l,v=r*h,p>=0)if(u>=-v)if(u<=v){let M=1/h;p*=M,u*=M,d=p*(p+a*u+2*o)+u*(a*p+u+2*l)+c}else u=r,p=Math.max(0,-(a*u+o)),d=-p*p+u*(u+2*l)+c;else u=-r,p=Math.max(0,-(a*u+o)),d=-p*p+u*(u+2*l)+c;else u<=-v?(p=Math.max(0,-(-a*r+o)),u=p>0?-r:Math.min(Math.max(-r,-l),r),d=-p*p+u*(u+2*l)+c):u<=v?(p=0,u=Math.min(Math.max(-r,-l),r),d=u*(u+2*l)+c):(p=Math.max(0,-(a*r+o)),u=p>0?r:Math.min(Math.max(-r,-l),r),d=-p*p+u*(u+2*l)+c);else u=a>0?-r:r,p=Math.max(0,-(a*u+o)),d=-p*p+u*(u+2*l)+c;return i&&i.copy(this.origin).addScaledVector(this.direction,p),s&&s.copy(Zg).addScaledVector(Pf,u),d}intersectSphere(e,n){if(e.radius<0)return null;zs.subVectors(e.center,this.origin);let i=zs.dot(this.direction),s=zs.dot(zs)-i*i,r=e.radius*e.radius;if(s>r)return null;let a=Math.sqrt(r-s),o=i-a,l=i+a;return l<0?null:o<0?this.at(l,n):this.at(o,n)}intersectsSphere(e){return e.radius<0?!1:this.distanceSqToPoint(e.center)<=e.radius*e.radius}distanceToPlane(e){let n=e.normal.dot(this.direction);if(n===0)return e.distanceToPoint(this.origin)===0?0:null;let i=-(this.origin.dot(e.normal)+e.constant)/n;return i>=0?i:null}intersectPlane(e,n){let i=this.distanceToPlane(e);return i===null?null:this.at(i,n)}intersectsPlane(e){let n=e.distanceToPoint(this.origin);return n===0||e.normal.dot(this.direction)*n<0}intersectBox(e,n){let i,s,r,a,o,l,c=1/this.direction.x,h=1/this.direction.y,p=1/this.direction.z,u=this.origin;return c>=0?(i=(e.min.x-u.x)*c,s=(e.max.x-u.x)*c):(i=(e.max.x-u.x)*c,s=(e.min.x-u.x)*c),h>=0?(r=(e.min.y-u.y)*h,a=(e.max.y-u.y)*h):(r=(e.max.y-u.y)*h,a=(e.min.y-u.y)*h),i>a||r>s||((r>i||isNaN(i))&&(i=r),(a<s||isNaN(s))&&(s=a),p>=0?(o=(e.min.z-u.z)*p,l=(e.max.z-u.z)*p):(o=(e.max.z-u.z)*p,l=(e.min.z-u.z)*p),i>l||o>s)||((o>i||i!==i)&&(i=o),(l<s||s!==s)&&(s=l),s<0)?null:this.at(i>=0?i:s,n)}intersectsBox(e){return this.intersectBox(e,zs)!==null}intersectTriangle(e,n,i,s,r){let a=this.origin,o=this.direction,l=o.x,c=o.y,h=o.z,p=e.x-a.x,u=e.y-a.y,d=e.z-a.z,v=n.x-a.x,M=n.y-a.y,m=n.z-a.z,f=i.x-a.x,g=i.y-a.y,S=i.z-a.z,_=Math.abs(l),E=Math.abs(c),T=Math.abs(h),C,y,b,R,N,F,k,B,z,Z,q,ie;if(_>=E&&_>=T?(b=l,F=p,z=v,ie=f,l>=0?(C=c,y=h,R=u,N=d,k=M,B=m,Z=g,q=S):(C=h,y=c,R=d,N=u,k=m,B=M,Z=S,q=g)):E>=T?(b=c,F=u,z=M,ie=g,c>=0?(C=h,y=l,R=d,N=p,k=m,B=v,Z=S,q=f):(C=l,y=h,R=p,N=d,k=v,B=m,Z=f,q=S)):(b=h,F=d,z=m,ie=S,h>=0?(C=l,y=c,R=p,N=u,k=v,B=M,Z=f,q=g):(C=c,y=l,R=u,N=p,k=M,B=v,Z=g,q=f)),b===0)return null;let W=C/b,$=y/b,te=1/b,we=R-W*F,Me=N-$*F,ut=k-W*z,qe=B-$*z,$e=Z-W*ie,X=q-$*ie,ee=$e*qe-X*ut,ue=we*X-Me*$e,Se=ut*Me-qe*we;if(s){if(ee<0||ue<0||Se<0)return null}else if((ee<0||ue<0||Se<0)&&(ee>0||ue>0||Se>0))return null;let ge=ee+ue+Se;if(ge===0)return null;let Fe=te*(ee*F+ue*z+Se*ie);return(ge>0?Fe<0:Fe>0)?null:this.at(Fe/ge,r)}applyMatrix4(e){return this.origin.applyMatrix4(e),this.direction.transformDirection(e),this}equals(e){return e.origin.equals(this.origin)&&e.direction.equals(this.direction)}clone(){return new this.constructor().copy(this)}},vc=class extends Vi{constructor(e){super(),this.isMeshBasicMaterial=!0,this.type="MeshBasicMaterial",this.color=new Ye(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new Rr,this.combine=u0,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.specularMap=e.specularMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.combine=e.combine,this.reflectivity=e.reflectivity,this.refractionRatio=e.refractionRatio,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.fog=e.fog,this}},ZA=new kt,ma=new lh,Of=new Yo,KA=new G,Ff=new G,zf=new G,Hf=new G,Kg=new G,Gf=new G,JA=new G,Vf=new G,Rn=class extends bi{constructor(e=new wi,n=new vc){super(),this.isMesh=!0,this.type="Mesh",this.geometry=e,this.material=n,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.count=1,this.updateMorphTargets()}copy(e,n){return super.copy(e,n),e.morphTargetInfluences!==void 0&&(this.morphTargetInfluences=e.morphTargetInfluences.slice()),e.morphTargetDictionary!==void 0&&(this.morphTargetDictionary=Object.assign({},e.morphTargetDictionary)),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}updateMorphTargets(){let n=this.geometry.morphAttributes,i=Object.keys(n);if(i.length>0){let s=n[i[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,a=s.length;r<a;r++){let o=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[o]=r}}}}getVertexPosition(e,n){let i=this.geometry,s=i.attributes.position,r=i.morphAttributes.position,a=i.morphTargetsRelative;n.fromBufferAttribute(s,e);let o=this.morphTargetInfluences;if(r&&o){Gf.set(0,0,0);for(let l=0,c=r.length;l<c;l++){let h=o[l],p=r[l];h!==0&&(Kg.fromBufferAttribute(p,e),a?Gf.addScaledVector(Kg,h):Gf.addScaledVector(Kg.sub(n),h))}n.add(Gf)}return n}intersectsFrustum(e){return e.intersectsObject(this)}raycast(e,n){let i=this.geometry,s=this.material,r=this.matrixWorld;s!==void 0&&(i.boundingSphere===null&&i.computeBoundingSphere(),Of.copy(i.boundingSphere),Of.applyMatrix4(r),ma.copy(e.ray).recast(e.near),!(Of.containsPoint(ma.origin)===!1&&(ma.intersectSphere(Of,KA)===null||ma.origin.distanceToSquared(KA)>(e.far-e.near)**2))&&(ZA.copy(r).invert(),ma.copy(e.ray).applyMatrix4(ZA),!(i.boundingBox!==null&&ma.intersectsBox(i.boundingBox)===!1)&&this._computeIntersections(e,n,ma)))}_computeIntersections(e,n,i){let s,r=this.geometry,a=this.material,o=r.index,l=r.attributes.position,c=r.attributes.uv,h=r.attributes.uv1,p=r.attributes.normal,u=r.groups,d=r.drawRange;if(o!==null)if(Array.isArray(a))for(let v=0,M=u.length;v<M;v++){let m=u[v],f=a[m.materialIndex],g=Math.max(m.start,d.start),S=Math.min(o.count,Math.min(m.start+m.count,d.start+d.count));for(let _=g,E=S;_<E;_+=3){let T=o.getX(_),C=o.getX(_+1),y=o.getX(_+2);s=kf(this,f,e,i,c,h,p,T,C,y),s&&(s.faceIndex=Math.floor(_/3),s.face.materialIndex=m.materialIndex,n.push(s))}}else{let v=Math.max(0,d.start),M=Math.min(o.count,d.start+d.count);for(let m=v,f=M;m<f;m+=3){let g=o.getX(m),S=o.getX(m+1),_=o.getX(m+2);s=kf(this,a,e,i,c,h,p,g,S,_),s&&(s.faceIndex=Math.floor(m/3),n.push(s))}}else if(l!==void 0)if(Array.isArray(a))for(let v=0,M=u.length;v<M;v++){let m=u[v],f=a[m.materialIndex],g=Math.max(m.start,d.start),S=Math.min(l.count,Math.min(m.start+m.count,d.start+d.count));for(let _=g,E=S;_<E;_+=3){let T=_,C=_+1,y=_+2;s=kf(this,f,e,i,c,h,p,T,C,y),s&&(s.faceIndex=Math.floor(_/3),s.face.materialIndex=m.materialIndex,n.push(s))}}else{let v=Math.max(0,d.start),M=Math.min(l.count,d.start+d.count);for(let m=v,f=M;m<f;m+=3){let g=m,S=m+1,_=m+2;s=kf(this,a,e,i,c,h,p,g,S,_),s&&(s.faceIndex=Math.floor(m/3),n.push(s))}}}};function Bw(t,e,n,i,s,r,a,o){let l;if(e.side===$t?l=i.intersectTriangle(a,r,s,!0,o):l=i.intersectTriangle(s,r,a,e.side===ps,o),l===null)return null;Vf.copy(o),Vf.applyMatrix4(t.matrixWorld);let c=n.ray.origin.distanceTo(Vf);return c<n.near||c>n.far?null:{distance:c,point:Vf.clone(),object:t}}function kf(t,e,n,i,s,r,a,o,l,c){t.getVertexPosition(o,Ff),t.getVertexPosition(l,zf),t.getVertexPosition(c,Hf);let h=Bw(t,e,n,i,Ff,zf,Hf,JA);if(h){let p=new G;Cr.getBarycoord(JA,Ff,zf,Hf,p),s&&(h.uv=Cr.getInterpolatedAttribute(s,o,l,c,p,new Ne)),r&&(h.uv1=Cr.getInterpolatedAttribute(r,o,l,c,p,new Ne)),a&&(h.normal=Cr.getInterpolatedAttribute(a,o,l,c,p,new G),h.normal.dot(i.direction)>0&&h.normal.multiplyScalar(-1));let u={a:o,b:l,c,normal:new G,materialIndex:0};Cr.getNormal(Ff,zf,Hf,u.normal),h.face=u,h.barycoord=p}return h}var ch=class extends jt{constructor(e=null,n=1,i=1,s,r,a,o,l,c=dn,h=dn,p,u){super(null,a,o,l,c,h,s,r,p,u),this.isDataTexture=!0,this.image={data:e,width:n,height:i},this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}};var ga=new Yo,Nw=new Ne(.5,.5),Wf=new G,xc=class{constructor(e=new Hi,n=new Hi,i=new Hi,s=new Hi,r=new Hi,a=new Hi){this.planes=[e,n,i,s,r,a]}set(e,n,i,s,r,a){let o=this.planes;return o[0].copy(e),o[1].copy(n),o[2].copy(i),o[3].copy(s),o[4].copy(r),o[5].copy(a),this}copy(e){let n=this.planes;for(let i=0;i<6;i++)n[i].copy(e.planes[i]);return this}setFromProjectionMatrix(e,n=Gi,i=!1){let s=this.planes,r=e.elements,a=r[0],o=r[1],l=r[2],c=r[3],h=r[4],p=r[5],u=r[6],d=r[7],v=r[8],M=r[9],m=r[10],f=r[11],g=r[12],S=r[13],_=r[14],E=r[15];if(s[0].setComponents(c-a,d-h,f-v,E-g).normalize(),s[1].setComponents(c+a,d+h,f+v,E+g).normalize(),s[2].setComponents(c+o,d+p,f+M,E+S).normalize(),s[3].setComponents(c-o,d-p,f-M,E-S).normalize(),i)s[4].setComponents(l,u,m,_).normalize(),s[5].setComponents(c-l,d-u,f-m,E-_).normalize();else if(s[4].setComponents(c-l,d-u,f-m,E-_).normalize(),n===Gi)s[5].setComponents(c+l,d+u,f+m,E+_).normalize();else if(n===fc)s[5].setComponents(l,u,m,_).normalize();else throw new Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: "+n);return this}intersectsObject(e){if(e.boundingSphere!==void 0)e.boundingSphere===null&&e.computeBoundingSphere(),ga.copy(e.boundingSphere).applyMatrix4(e.matrixWorld);else{let n=e.geometry;n.boundingSphere===null&&n.computeBoundingSphere(),ga.copy(n.boundingSphere).applyMatrix4(e.matrixWorld)}return this.intersectsSphere(ga)}intersectsSprite(e){ga.center.set(0,0,0);let n=Nw.distanceTo(e.center);return ga.radius=.7071067811865476+n,ga.applyMatrix4(e.matrixWorld),this.intersectsSphere(ga)}intersectsSphere(e){let n=this.planes,i=e.center,s=-e.radius;for(let r=0;r<6;r++)if(n[r].distanceToPoint(i)<s)return!1;return!0}intersectsBox(e){let n=this.planes;for(let i=0;i<6;i++){let s=n[i];if(Wf.x=s.normal.x>0?e.max.x:e.min.x,Wf.y=s.normal.y>0?e.max.y:e.min.y,Wf.z=s.normal.z>0?e.max.z:e.min.z,s.distanceToPoint(Wf)<0)return!1}return!0}containsPoint(e){let n=this.planes;for(let i=0;i<6;i++)if(n[i].distanceToPoint(e)<0)return!1;return!0}clone(){return new this.constructor().copy(this)}};var yc=class extends jt{constructor(e=[],n=Pr,i,s,r,a,o,l,c,h){super(e,n,i,s,r,a,o,l,c,h),this.isCubeTexture=!0,this.flipY=!1}get images(){return this.image}set images(e){this.image=e}};var Ci=class extends jt{constructor(e,n,i=Wi,s,r,a,o=dn,l=dn,c,h=hs,p=1){if(h!==hs&&h!==ms)throw new Error("THREE.DepthTexture: format must be either THREE.DepthFormat or THREE.DepthStencilFormat");let u={width:e,height:n,depth:p};super(u,s,r,a,o,l,h,i,c),this.isDepthTexture=!0,this.flipY=!1,this.generateMipmaps=!1,this.compareFunction=null}copy(e){return super.copy(e),this.source=new Wo(Object.assign({},e.image)),this.compareFunction=e.compareFunction,this}toJSON(e){let n=super.toJSON(e);return n.compareFunction=this.compareFunction,n}},uh=class extends Ci{constructor(e,n=Wi,i=Pr,s,r,a=dn,o=dn,l,c=hs){let h={width:e,height:e,depth:1},p=[h,h,h,h,h,h];super(e,e,n,i,s,r,a,o,l,c),this.image=p,this.isCubeDepthTexture=!0,this.isCubeTexture=!0}get images(){return this.image}set images(e){this.image=e}},_c=class extends jt{constructor(e=null){super(),this.sourceTexture=e,this.isExternalTexture=!0}copy(e){return super.copy(e),this.sourceTexture=e.sourceTexture,this}},qo=class t extends wi{constructor(e=1,n=1,i=1,s=1,r=1,a=1){super(),this.type="BoxGeometry",this.parameters={width:e,height:n,depth:i,widthSegments:s,heightSegments:r,depthSegments:a};let o=this;s=Math.floor(s),r=Math.floor(r),a=Math.floor(a);let l=[],c=[],h=[],p=[],u=0,d=0;v("z","y","x",-1,-1,i,n,e,a,r,0),v("z","y","x",1,-1,i,n,-e,a,r,1),v("x","z","y",1,1,e,i,n,s,a,2),v("x","z","y",1,-1,e,i,-n,s,a,3),v("x","y","z",1,-1,e,n,i,s,r,4),v("x","y","z",-1,-1,e,n,-i,s,r,5),this.setIndex(l),this.setAttribute("position",new Ti(c,3)),this.setAttribute("normal",new Ti(h,3)),this.setAttribute("uv",new Ti(p,2));function v(M,m,f,g,S,_,E,T,C,y,b){let R=_/C,N=E/y,F=_/2,k=E/2,B=T/2,z=C+1,Z=y+1,q=0,ie=0,W=new G;for(let $=0;$<Z;$++){let te=$*N-k;for(let we=0;we<z;we++){let Me=we*R-F;W[M]=Me*g,W[m]=te*S,W[f]=B,c.push(W.x,W.y,W.z),W[M]=0,W[m]=0,W[f]=T>0?1:-1,h.push(W.x,W.y,W.z),p.push(we/C),p.push(1-$/y),q+=1}}for(let $=0;$<y;$++)for(let te=0;te<C;te++){let we=u+te+z*$,Me=u+te+z*($+1),ut=u+(te+1)+z*($+1),qe=u+(te+1)+z*$;l.push(we,Me,qe),l.push(Me,ut,qe),ie+=6}o.addGroup(d,ie,b),d+=ie,u+=q}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new t(e.width,e.height,e.depth,e.widthSegments,e.heightSegments,e.depthSegments)}};var ya=class t extends wi{constructor(e=1,n=1,i=1,s=1){super(),this.type="PlaneGeometry",this.parameters={width:e,height:n,widthSegments:i,heightSegments:s};let r=e/2,a=n/2,o=Math.floor(i),l=Math.floor(s),c=o+1,h=l+1,p=e/o,u=n/l,d=[],v=[],M=[],m=[];for(let f=0;f<h;f++){let g=f*u-a;for(let S=0;S<c;S++){let _=S*p-r;v.push(_,-g,0),M.push(0,0,1),m.push(S/o),m.push(1-f/l)}}for(let f=0;f<l;f++)for(let g=0;g<o;g++){let S=g+c*f,_=g+c*(f+1),E=g+1+c*(f+1),T=g+1+c*f;d.push(S,_,T),d.push(_,E,T)}this.setIndex(d),this.setAttribute("position",new Ti(v,3)),this.setAttribute("normal",new Ti(M,3)),this.setAttribute("uv",new Ti(m,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new t(e.width,e.height,e.widthSegments,e.heightSegments)}};function Aa(t){let e={};for(let n in t){e[n]={};for(let i in t[n]){let s=t[n][i];if(jA(s))s.isRenderTargetTexture?(De("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."),e[n][i]=null):e[n][i]=s.clone();else if(Array.isArray(s))if(jA(s[0])){let r=[];for(let a=0,o=s.length;a<o;a++)r[a]=s[a].clone();e[n][i]=r}else e[n][i]=s.slice();else e[n][i]=s}}return e}function Dn(t){let e={};for(let n=0;n<t.length;n++){let i=Aa(t[n]);for(let s in i)e[s]=i[s]}return e}function jA(t){return t&&(t.isColor||t.isMatrix3||t.isMatrix4||t.isVector2||t.isVector3||t.isVector4||t.isTexture||t.isQuaternion)}function Pw(t){let e=[];for(let n=0;n<t.length;n++)e.push(t[n].clone());return e}function C0(t){let e=t.getRenderTarget();return e===null?t.outputColorSpace:e.isXRRenderTarget===!0?e.texture.colorSpace:Ke.workingColorSpace}var H1={clone:Aa,merge:Dn},Lw=`void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`,Ow=`void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`,Wt=class extends Vi{constructor(e){super(),this.isShaderMaterial=!0,this.type="ShaderMaterial",this.defines={},this.uniforms={},this.uniformsGroups=[],this.vertexShader=Lw,this.fragmentShader=Ow,this.linewidth=1,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.lights=!1,this.clipping=!1,this.forceSinglePass=!0,this.extensions={clipCullDistance:!1,multiDraw:!1},this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv1:[0,0]},this.index0AttributeName=void 0,this.uniformsNeedUpdate=!1,this.glslVersion=null,e!==void 0&&this.setValues(e)}copy(e){return super.copy(e),this.fragmentShader=e.fragmentShader,this.vertexShader=e.vertexShader,this.uniforms=Aa(e.uniforms),this.uniformsGroups=Pw(e.uniformsGroups),this.defines=Object.assign({},e.defines),this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.fog=e.fog,this.lights=e.lights,this.clipping=e.clipping,this.extensions=Object.assign({},e.extensions),this.glslVersion=e.glslVersion,this.defaultAttributeValues=Object.assign({},e.defaultAttributeValues),this.index0AttributeName=e.index0AttributeName,this.uniformsNeedUpdate=e.uniformsNeedUpdate,this}toJSON(e){let n=super.toJSON(e);n.glslVersion=this.glslVersion,n.uniforms={};for(let s in this.uniforms){let a=this.uniforms[s].value;a&&a.isTexture?n.uniforms[s]={type:"t",value:a.toJSON(e).uuid}:a&&a.isColor?n.uniforms[s]={type:"c",value:a.getHex()}:a&&a.isVector2?n.uniforms[s]={type:"v2",value:a.toArray()}:a&&a.isVector3?n.uniforms[s]={type:"v3",value:a.toArray()}:a&&a.isVector4?n.uniforms[s]={type:"v4",value:a.toArray()}:a&&a.isMatrix3?n.uniforms[s]={type:"m3",value:a.toArray()}:a&&a.isMatrix4?n.uniforms[s]={type:"m4",value:a.toArray()}:n.uniforms[s]={value:a}}Object.keys(this.defines).length>0&&(n.defines=this.defines),n.vertexShader=this.vertexShader,n.fragmentShader=this.fragmentShader,n.lights=this.lights,n.clipping=this.clipping;let i={};for(let s in this.extensions)this.extensions[s]===!0&&(i[s]=!0);return Object.keys(i).length>0&&(n.extensions=i),n}fromJSON(e,n){if(super.fromJSON(e,n),e.uniforms!==void 0)for(let i in e.uniforms){let s=e.uniforms[i];switch(this.uniforms[i]={},s.type){case"t":this.uniforms[i].value=n[s.value]||null;break;case"c":this.uniforms[i].value=new Ye().setHex(s.value);break;case"v2":this.uniforms[i].value=new Ne().fromArray(s.value);break;case"v3":this.uniforms[i].value=new G().fromArray(s.value);break;case"v4":this.uniforms[i].value=new Ot().fromArray(s.value);break;case"m3":this.uniforms[i].value=new Pe().fromArray(s.value);break;case"m4":this.uniforms[i].value=new kt().fromArray(s.value);break;default:this.uniforms[i].value=s.value}}if(e.defines!==void 0&&(this.defines=e.defines),e.vertexShader!==void 0&&(this.vertexShader=e.vertexShader),e.fragmentShader!==void 0&&(this.fragmentShader=e.fragmentShader),e.glslVersion!==void 0&&(this.glslVersion=e.glslVersion),e.extensions!==void 0)for(let i in e.extensions)this.extensions[i]=e.extensions[i];return e.lights!==void 0&&(this.lights=e.lights),e.clipping!==void 0&&(this.clipping=e.clipping),this}},fh=class extends Wt{constructor(e){super(e),this.isRawShaderMaterial=!0,this.type="RawShaderMaterial"}};var hh=class extends Vi{constructor(e){super(),this.isMeshDepthMaterial=!0,this.type="MeshDepthMaterial",this.depthPacking=gs,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.setValues(e)}copy(e){return super.copy(e),this.depthPacking=e.depthPacking,this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this}},dh=class extends Vi{constructor(e){super(),this.isMeshDistanceMaterial=!0,this.type="MeshDistanceMaterial",this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.setValues(e)}copy(e){return super.copy(e),this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this}};function Oo(t,e){return!t||t.constructor===e?t:typeof e.BYTES_PER_ELEMENT=="number"?new e(t):Array.prototype.slice.call(t)}function Jg(t){return t!==void 0&&t.inTangents!==void 0&&t.outTangents!==void 0}var Ir=class{constructor(e,n,i,s){this.parameterPositions=e,this._cachedIndex=0,this.resultBuffer=s!==void 0?s:new n.constructor(i),this.sampleValues=n,this.valueSize=i,this.settings=null,this.DefaultSettings_={}}evaluate(e){let n=this.parameterPositions,i=this._cachedIndex,s=n[i],r=n[i-1];e:{t:{let a;n:{i:if(!(e<s)){for(let o=i+2;;){if(s===void 0){if(e<r)break i;return i=n.length,this._cachedIndex=i,this.copySampleValue_(i-1)}if(i===o)break;if(r=s,s=n[++i],e<s)break t}a=n.length;break n}if(!(e>=r)){let o=n[1];e<o&&(i=2,r=o);for(let l=i-2;;){if(r===void 0)return this._cachedIndex=0,this.copySampleValue_(0);if(i===l)break;if(s=r,r=n[--i-1],e>=r)break t}a=i,i=0;break n}break e}for(;i<a;){let o=i+a>>>1;e<n[o]?a=o:i=o+1}if(s=n[i],r=n[i-1],r===void 0)return this._cachedIndex=0,this.copySampleValue_(0);if(s===void 0)return i=n.length,this._cachedIndex=i,this.copySampleValue_(i-1)}this._cachedIndex=i,this.intervalChanged_(i,r,s)}return this.interpolate_(i,r,e,s)}getSettings_(){return this.settings||this.DefaultSettings_}copySampleValue_(e){let n=this.resultBuffer,i=this.sampleValues,s=this.valueSize,r=e*s;for(let a=0;a!==s;++a)n[a]=i[r+a];return n}interpolate_(){throw new Error("THREE.Interpolant: Call to abstract method.")}intervalChanged_(){}},ph=class extends Ir{constructor(e,n,i,s){super(e,n,i,s),this._weightPrev=-0,this._offsetPrev=-0,this._weightNext=-0,this._offsetNext=-0,this.DefaultSettings_={endingStart:$g,endingEnd:$g}}intervalChanged_(e,n,i){let s=this.parameterPositions,r=e-2,a=e+1,o=s[r],l=s[a];if(o===void 0)switch(this.getSettings_().endingStart){case e0:r=e,o=2*n-i;break;case t0:r=s.length-2,o=n+s[r]-s[r+1];break;default:r=e,o=i}if(l===void 0)switch(this.getSettings_().endingEnd){case e0:a=e,l=2*i-n;break;case t0:a=1,l=i+s[1]-s[0];break;default:a=e-1,l=n}let c=(i-n)*.5,h=this.valueSize;this._weightPrev=c/(n-o),this._weightNext=c/(l-i),this._offsetPrev=r*h,this._offsetNext=a*h}interpolate_(e,n,i,s){let r=this.resultBuffer,a=this.sampleValues,o=this.valueSize,l=e*o,c=l-o,h=this._offsetPrev,p=this._offsetNext,u=this._weightPrev,d=this._weightNext,v=(i-n)/(s-n),M=v*v,m=M*v,f=-u*m+2*u*M-u*v,g=(1+u)*m+(-1.5-2*u)*M+(-.5+u)*v+1,S=(-1-d)*m+(1.5+d)*M+.5*v,_=d*m-d*M;for(let E=0;E!==o;++E)r[E]=f*a[h+E]+g*a[c+E]+S*a[l+E]+_*a[p+E];return r}},mh=class extends Ir{constructor(e,n,i,s){super(e,n,i,s)}interpolate_(e,n,i,s){let r=this.resultBuffer,a=this.sampleValues,o=this.valueSize,l=e*o,c=l-o,h=(i-n)/(s-n),p=1-h;for(let u=0;u!==o;++u)r[u]=a[c+u]*p+a[l+u]*h;return r}},gh=class extends Ir{constructor(e,n,i,s){super(e,n,i,s)}interpolate_(e){return this.copySampleValue_(e-1)}},vh=class extends Ir{interpolate_(e,n,i,s){let r=this.resultBuffer,a=this.sampleValues,o=this.valueSize,l=e*o,c=l-o,h=this.inTangents,p=this.outTangents;if(!h||!p){let v=(i-n)/(s-n),M=1-v;for(let m=0;m!==o;++m)r[m]=a[c+m]*M+a[l+m]*v;return r}let u=o*2,d=e-1;for(let v=0;v!==o;++v){let M=a[c+v],m=a[l+v],f=d*u+v*2,g=p[f],S=p[f+1],_=e*u+v*2,E=h[_],T=h[_+1],C=zw(i,n,g,E,s);r[v]=G1(C,M,S,T,m)}return r}};function G1(t,e,n,i,s){let r=1-t;return r*r*r*e+3*r*r*t*n+3*r*t*t*i+t*t*t*s}function Fw(t,e,n,i,s){let r=1-t;return 3*r*r*(n-e)+6*r*t*(i-n)+3*t*t*(s-i)}function zw(t,e,n,i,s){let r=(t-e)/(s-e);for(let a=0;a<8;a++){let o=G1(r,e,n,i,s)-t;if(Math.abs(o)<1e-10)break;let l=Fw(r,e,n,i,s);if(Math.abs(l)<1e-10)break;r=Math.max(0,Math.min(1,r-o/l))}return r}var ui=class{constructor(e,n,i,s){if(e===void 0)throw new Error("THREE.KeyframeTrack: track name is undefined");if(n===void 0||n.length===0)throw new Error("THREE.KeyframeTrack: no keyframes in track named "+e);this.name=e,this.times=Oo(n,this.TimeBufferType),this.values=Oo(i,this.ValueBufferType),this.setInterpolation(s||this.DefaultInterpolation)}static toJSON(e){let n=e.constructor,i;if(n.toJSON!==this.toJSON)i=n.toJSON(e);else{i={name:e.name,times:Oo(e.times,Array),values:Oo(e.values,Array)};let s=e.getInterpolation();s!==e.DefaultInterpolation&&(i.interpolation=s),Jg(e.settings)&&(i.settings={inTangents:Oo(e.settings.inTangents,Array),outTangents:Oo(e.settings.outTangents,Array)})}return i.type=e.ValueTypeName,i}InterpolantFactoryMethodDiscrete(e){return new gh(this.times,this.values,this.getValueSize(),e)}InterpolantFactoryMethodLinear(e){return new mh(this.times,this.values,this.getValueSize(),e)}InterpolantFactoryMethodSmooth(e){return new ph(this.times,this.values,this.getValueSize(),e)}InterpolantFactoryMethodBezier(e){let n=new vh(this.times,this.values,this.getValueSize(),e);return this.settings&&(n.inTangents=this.settings.inTangents,n.outTangents=this.settings.outTangents),n}setInterpolation(e){let n;switch(e){case cc:n=this.InterpolantFactoryMethodDiscrete;break;case ih:n=this.InterpolantFactoryMethodLinear;break;case qf:n=this.InterpolantFactoryMethodSmooth;break;case jg:n=this.InterpolantFactoryMethodBezier;break}if(n===void 0){let i="unsupported interpolation for "+this.ValueTypeName+" keyframe track named "+this.name;if(this.createInterpolant===void 0)if(e!==this.DefaultInterpolation)this.setInterpolation(this.DefaultInterpolation);else throw new Error(i);return De("KeyframeTrack:",i),this}return this.createInterpolant=n,this}getInterpolation(){switch(this.createInterpolant){case this.InterpolantFactoryMethodDiscrete:return cc;case this.InterpolantFactoryMethodLinear:return ih;case this.InterpolantFactoryMethodSmooth:return qf;case this.InterpolantFactoryMethodBezier:return jg}}getValueSize(){return this.values.length/this.times.length}shift(e){if(e!==0){let n=this.times;for(let i=0,s=n.length;i!==s;++i)n[i]+=e}return this}scale(e){if(e!==1){let n=this.times;for(let i=0,s=n.length;i!==s;++i)n[i]*=e;Jg(this.settings)&&($A(this.settings.inTangents,e),$A(this.settings.outTangents,e))}return this}trim(e,n){let i=this.times,s=i.length,r=0,a=s-1;for(;r!==s&&i[r]<e;)++r;for(;a!==-1&&i[a]>n;)--a;if(++a,r!==0||a!==s){r>=a&&(a=Math.max(a,1),r=a-1);let o=this.getValueSize();this.times=i.slice(r,a),this.values=this.values.slice(r*o,a*o)}return this}validate(){let e=!0,n=this.getValueSize();n-Math.floor(n)!==0&&(Ie("KeyframeTrack: Invalid value size in track.",this),e=!1);let i=this.times,s=this.values,r=i.length;r===0&&(Ie("KeyframeTrack: Track is empty.",this),e=!1);let a=null;for(let o=0;o!==r;o++){let l=i[o];if(typeof l=="number"&&isNaN(l)){Ie("KeyframeTrack: Time is not a valid number.",this,o,l),e=!1;break}if(a!==null&&a>l){Ie("KeyframeTrack: Out of order keys.",this,o,l,a),e=!1;break}a=l}if(s!==void 0&&mw(s))for(let o=0,l=s.length;o!==l;++o){let c=s[o];if(isNaN(c)){Ie("KeyframeTrack: Value is not a valid number.",this,o,c),e=!1;break}}return e}optimize(){let e=this.times.slice(),n=this.values.slice(),i=this.getValueSize(),s=this.getInterpolation()===qf,r=e.length-1,a=1;for(let o=1;o<r;++o){let l=!1,c=e[o],h=e[o+1];if(c!==h&&(o!==1||c!==e[0]))if(s)l=!0;else{let p=o*i,u=p-i,d=p+i;for(let v=0;v!==i;++v){let M=n[p+v];if(M!==n[u+v]||M!==n[d+v]){l=!0;break}}}if(l){if(o!==a){e[a]=e[o];let p=o*i,u=a*i;for(let d=0;d!==i;++d)n[u+d]=n[p+d]}++a}}if(r>0){e[a]=e[r];for(let o=r*i,l=a*i,c=0;c!==i;++c)n[l+c]=n[o+c];++a}return a!==e.length?(this.times=e.slice(0,a),this.values=n.slice(0,a*i)):(this.times=e,this.values=n),this}clone(){let e=this.times.slice(),n=this.values.slice(),i=this.constructor,s=new i(this.name,e,n);return s.createInterpolant=this.createInterpolant,Jg(this.settings)&&(s.settings={inTangents:this.settings.inTangents.slice(),outTangents:this.settings.outTangents.slice()}),s}};function $A(t,e){for(let n=0,i=t.length;n!==i;n+=2)t[n]*=e}ui.prototype.ValueTypeName="";ui.prototype.TimeBufferType=Float32Array;ui.prototype.ValueBufferType=Float32Array;ui.prototype.DefaultInterpolation=ih;var Br=class extends ui{constructor(e,n,i){super(e,n,i)}};Br.prototype.ValueTypeName="bool";Br.prototype.ValueBufferType=Array;Br.prototype.DefaultInterpolation=cc;Br.prototype.InterpolantFactoryMethodLinear=void 0;Br.prototype.InterpolantFactoryMethodSmooth=void 0;var xh=class extends ui{constructor(e,n,i,s){super(e,n,i,s)}};xh.prototype.ValueTypeName="color";var yh=class extends ui{constructor(e,n,i,s){super(e,n,i,s)}};yh.prototype.ValueTypeName="number";var _h=class extends Ir{constructor(e,n,i,s){super(e,n,i,s)}interpolate_(e,n,i,s){let r=this.resultBuffer,a=this.sampleValues,o=this.valueSize,l=(i-n)/(s-n),c=e*o;for(let h=c+o;c!==h;c+=4)ds.slerpFlat(r,0,a,c-o,a,c,l);return r}},Sc=class extends ui{constructor(e,n,i,s){super(e,n,i,s)}InterpolantFactoryMethodLinear(e){return new _h(this.times,this.values,this.getValueSize(),e)}};Sc.prototype.ValueTypeName="quaternion";Sc.prototype.InterpolantFactoryMethodSmooth=void 0;var Nr=class extends ui{constructor(e,n,i){super(e,n,i)}};Nr.prototype.ValueTypeName="string";Nr.prototype.ValueBufferType=Array;Nr.prototype.DefaultInterpolation=cc;Nr.prototype.InterpolantFactoryMethodLinear=void 0;Nr.prototype.InterpolantFactoryMethodSmooth=void 0;var Sh=class extends ui{constructor(e,n,i,s){super(e,n,i,s)}};Sh.prototype.ValueTypeName="vector";var Ah=class{constructor(e,n,i){let s=this,r=!1,a=0,o=0,l,c=[];this.onStart=void 0,this.onLoad=e,this.onProgress=n,this.onError=i,this._abortController=null,this.itemStart=function(h){o++,r===!1&&s.onStart!==void 0&&s.onStart(h,a,o),r=!0},this.itemEnd=function(h){a++,s.onProgress!==void 0&&s.onProgress(h,a,o),a===o&&(r=!1,s.onLoad!==void 0&&s.onLoad())},this.itemError=function(h){s.onError!==void 0&&s.onError(h)},this.resolveURL=function(h){return h=h.normalize("NFC"),l?l(h):h},this.setURLModifier=function(h){return l=h,this},this.addHandler=function(h,p){return c.push(h,p),this},this.removeHandler=function(h){let p=c.indexOf(h);return p!==-1&&c.splice(p,2),this},this.getHandler=function(h){for(let p=0,u=c.length;p<u;p+=2){let d=c[p],v=c[p+1];if(d.global&&(d.lastIndex=0),d.test(h))return v}return null},this.abort=function(){return this.abortController.abort(),this._abortController=null,this}}get abortController(){return this._abortController||(this._abortController=new AbortController),this._abortController}},V1=new Ah,Mh=class{constructor(e){this.manager=e!==void 0?e:V1,this.crossOrigin="anonymous",this.withCredentials=!1,this.path="",this.resourcePath="",this.requestHeader={},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}load(){}loadAsync(e,n){let i=this;return new Promise(function(s,r){i.load(e,s,n,r)})}parse(){}setCrossOrigin(e){return this.crossOrigin=e,this}setWithCredentials(e){return this.withCredentials=e,this}setPath(e){return this.path=e,this}setResourcePath(e){return this.resourcePath=e,this}setRequestHeader(e){return this.requestHeader=e,this}abort(){return this}};Mh.DEFAULT_MATERIAL_NAME="__DEFAULT";var Xf=new G,Yf=new ds,us=new G,Ac=class extends bi{constructor(){super(),this.isCamera=!0,this.type="Camera",this.matrixWorldInverse=new kt,this.projectionMatrix=new kt,this.projectionMatrixInverse=new kt,this.coordinateSystem=Gi,this._reversedDepth=!1}get reversedDepth(){return this._reversedDepth}copy(e,n){return super.copy(e,n),this.matrixWorldInverse.copy(e.matrixWorldInverse),this.projectionMatrix.copy(e.projectionMatrix),this.projectionMatrixInverse.copy(e.projectionMatrixInverse),this.coordinateSystem=e.coordinateSystem,this}getWorldDirection(e){return super.getWorldDirection(e).negate()}updateMatrixWorld(e){super.updateMatrixWorld(e),this.matrixWorld.decompose(Xf,Yf,us),us.x===1&&us.y===1&&us.z===1?this.matrixWorldInverse.copy(this.matrixWorld).invert():this.matrixWorldInverse.compose(Xf,Yf,us.set(1,1,1)).invert()}updateWorldMatrix(e,n,i=!1){super.updateWorldMatrix(e,n,i),this.matrixWorld.decompose(Xf,Yf,us),us.x===1&&us.y===1&&us.z===1?this.matrixWorldInverse.copy(this.matrixWorld).invert():this.matrixWorldInverse.compose(Xf,Yf,us.set(1,1,1)).invert()}clone(){return new this.constructor().copy(this)}},wr=new G,e1=new Ne,t1=new Ne,Cn=class extends Ac{constructor(e=50,n=1,i=.1,s=2e3){super(),this.isPerspectiveCamera=!0,this.type="PerspectiveCamera",this.fov=e,this.zoom=1,this.near=i,this.far=s,this.focus=10,this.aspect=n,this.view=null,this.filmGauge=35,this.filmOffset=0,this.updateProjectionMatrix()}copy(e,n){return super.copy(e,n),this.fov=e.fov,this.zoom=e.zoom,this.near=e.near,this.far=e.far,this.focus=e.focus,this.aspect=e.aspect,this.view=e.view===null?null:Object.assign({},e.view),this.filmGauge=e.filmGauge,this.filmOffset=e.filmOffset,this}setFocalLength(e){let n=.5*this.getFilmHeight()/e;this.fov=sh*2*Math.atan(n),this.updateProjectionMatrix()}getFocalLength(){let e=Math.tan(Dg*.5*this.fov);return .5*this.getFilmHeight()/e}getEffectiveFOV(){return sh*2*Math.atan(Math.tan(Dg*.5*this.fov)/this.zoom)}getFilmWidth(){return this.filmGauge*Math.min(this.aspect,1)}getFilmHeight(){return this.filmGauge/Math.max(this.aspect,1)}getViewBounds(e,n,i){wr.set(-1,-1,.5).applyMatrix4(this.projectionMatrixInverse),n.set(wr.x,wr.y).multiplyScalar(-e/wr.z),wr.set(1,1,.5).applyMatrix4(this.projectionMatrixInverse),i.set(wr.x,wr.y).multiplyScalar(-e/wr.z)}getViewSize(e,n){return this.getViewBounds(e,e1,t1),n.subVectors(t1,e1)}setViewOffset(e,n,i,s,r,a){this.aspect=e/n,this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=n,this.view.offsetX=i,this.view.offsetY=s,this.view.width=r,this.view.height=a,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){let e=this.near,n=e*Math.tan(Dg*.5*this.fov)/this.zoom,i=2*n,s=this.aspect*i,r=-.5*s,a=this.view;if(this.view!==null&&this.view.enabled){let l=a.fullWidth,c=a.fullHeight;r+=a.offsetX*s/l,n-=a.offsetY*i/c,s*=a.width/l,i*=a.height/c}let o=this.filmOffset;o!==0&&(r+=e*o/this.getFilmWidth()),this.projectionMatrix.makePerspective(r,r+s,n,n-i,e,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){let n=super.toJSON(e);return n.object.fov=this.fov,n.object.zoom=this.zoom,n.object.near=this.near,n.object.far=this.far,n.object.focus=this.focus,n.object.aspect=this.aspect,this.view!==null&&(n.object.view=Object.assign({},this.view)),n.object.filmGauge=this.filmGauge,n.object.filmOffset=this.filmOffset,n}};var Vs=class extends Ac{constructor(e=-1,n=1,i=1,s=-1,r=.1,a=2e3){super(),this.isOrthographicCamera=!0,this.type="OrthographicCamera",this.zoom=1,this.view=null,this.left=e,this.right=n,this.top=i,this.bottom=s,this.near=r,this.far=a,this.updateProjectionMatrix()}copy(e,n){return super.copy(e,n),this.left=e.left,this.right=e.right,this.top=e.top,this.bottom=e.bottom,this.near=e.near,this.far=e.far,this.zoom=e.zoom,this.view=e.view===null?null:Object.assign({},e.view),this}setViewOffset(e,n,i,s,r,a){this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=n,this.view.offsetX=i,this.view.offsetY=s,this.view.width=r,this.view.height=a,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){let e=(this.right-this.left)/(2*this.zoom),n=(this.top-this.bottom)/(2*this.zoom),i=(this.right+this.left)/2,s=(this.top+this.bottom)/2,r=i-e,a=i+e,o=s+n,l=s-n;if(this.view!==null&&this.view.enabled){let c=(this.right-this.left)/this.view.fullWidth/this.zoom,h=(this.top-this.bottom)/this.view.fullHeight/this.zoom;r+=c*this.view.offsetX,a=r+c*this.view.width,o-=h*this.view.offsetY,l=o-h*this.view.height}this.projectionMatrix.makeOrthographic(r,a,o,l,this.near,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){let n=super.toJSON(e);return n.object.zoom=this.zoom,n.object.left=this.left,n.object.right=this.right,n.object.top=this.top,n.object.bottom=this.bottom,n.object.near=this.near,n.object.far=this.far,this.view!==null&&(n.object.view=Object.assign({},this.view)),n}};var Fo=-90,zo=1,Eh=class extends bi{constructor(e,n,i){super(),this.type="CubeCamera",this.renderTarget=i,this.coordinateSystem=null,this.activeMipmapLevel=0;let s=new Cn(Fo,zo,e,n);s.layers=this.layers,this.add(s);let r=new Cn(Fo,zo,e,n);r.layers=this.layers,this.add(r);let a=new Cn(Fo,zo,e,n);a.layers=this.layers,this.add(a);let o=new Cn(Fo,zo,e,n);o.layers=this.layers,this.add(o);let l=new Cn(Fo,zo,e,n);l.layers=this.layers,this.add(l);let c=new Cn(Fo,zo,e,n);c.layers=this.layers,this.add(c)}updateCoordinateSystem(){let e=this.coordinateSystem,n=this.children.concat(),[i,s,r,a,o,l]=n;for(let c of n)this.remove(c);if(e===Gi)i.up.set(0,1,0),i.lookAt(1,0,0),s.up.set(0,1,0),s.lookAt(-1,0,0),r.up.set(0,0,-1),r.lookAt(0,1,0),a.up.set(0,0,1),a.lookAt(0,-1,0),o.up.set(0,1,0),o.lookAt(0,0,1),l.up.set(0,1,0),l.lookAt(0,0,-1);else if(e===fc)i.up.set(0,-1,0),i.lookAt(-1,0,0),s.up.set(0,-1,0),s.lookAt(1,0,0),r.up.set(0,0,1),r.lookAt(0,1,0),a.up.set(0,0,-1),a.lookAt(0,-1,0),o.up.set(0,-1,0),o.lookAt(0,0,1),l.up.set(0,-1,0),l.lookAt(0,0,-1);else throw new Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: "+e);for(let c of n)this.add(c),c.updateMatrixWorld()}update(e,n){this.parent===null&&this.updateMatrixWorld();let{renderTarget:i,activeMipmapLevel:s}=this;this.coordinateSystem!==e.coordinateSystem&&(this.coordinateSystem=e.coordinateSystem,this.updateCoordinateSystem());let[r,a,o,l,c,h]=this.children,p=e.getRenderTarget(),u=e.getActiveCubeFace(),d=e.getActiveMipmapLevel(),v=e.xr.enabled;e.xr.enabled=!1;let M=i.texture.generateMipmaps;i.texture.generateMipmaps=!1;let m=!1;e.isWebGLRenderer===!0?m=e.state.buffers.depth.getReversed():m=e.reversedDepthBuffer,e.setRenderTarget(i,0,s),m&&e.autoClear===!1&&e.clearDepth(),e.render(n,r),e.setRenderTarget(i,1,s),m&&e.autoClear===!1&&e.clearDepth(),e.render(n,a),e.setRenderTarget(i,2,s),m&&e.autoClear===!1&&e.clearDepth(),e.render(n,o),e.setRenderTarget(i,3,s),m&&e.autoClear===!1&&e.clearDepth(),e.render(n,l),e.setRenderTarget(i,4,s),m&&e.autoClear===!1&&e.clearDepth(),e.render(n,c),i.texture.generateMipmaps=M,e.setRenderTarget(i,5,s),m&&e.autoClear===!1&&e.clearDepth(),e.render(n,h),e.setRenderTarget(p,u,d),e.xr.enabled=v,i.texture.needsPMREMUpdate=!0}},Th=class extends Cn{constructor(e=[]){super(),this.isArrayCamera=!0,this.isMultiViewCamera=!1,this.cameras=e}};var R0="\\[\\]\\.:\\/",Hw=new RegExp("["+R0+"]","g"),D0="[^"+R0+"]",Gw="[^"+R0.replace("\\.","")+"]",Vw=/((?:WC+[\/:])*)/.source.replace("WC",D0),kw=/(WCOD+)?/.source.replace("WCOD",Gw),Ww=/(?:\.(WC+)(?:\[(.+)\])?)?/.source.replace("WC",D0),Xw=/\.(WC+)(?:\[(.+)\])?/.source.replace("WC",D0),Yw=new RegExp("^"+Vw+kw+Ww+Xw+"$"),qw=["material","materials","bones","map"],n0=class{constructor(e,n,i){let s=i||Rt.parseTrackName(n);this._targetGroup=e,this._bindings=e.subscribe_(n,s)}getValue(e,n){this.bind();let i=this._targetGroup.nCachedObjects_,s=this._bindings[i];s!==void 0&&s.getValue(e,n)}setValue(e,n){let i=this._bindings;for(let s=this._targetGroup.nCachedObjects_,r=i.length;s!==r;++s)i[s].setValue(e,n)}bind(){let e=this._bindings;for(let n=this._targetGroup.nCachedObjects_,i=e.length;n!==i;++n)e[n].bind()}unbind(){let e=this._bindings;for(let n=this._targetGroup.nCachedObjects_,i=e.length;n!==i;++n)e[n].unbind()}},Rt=class t{constructor(e,n,i){this.path=n,this.parsedPath=i||t.parseTrackName(n),this.node=t.findNode(e,this.parsedPath.nodeName),this.rootNode=e,this.getValue=this._getValue_unbound,this.setValue=this._setValue_unbound}static create(e,n,i){return e&&e.isAnimationObjectGroup?new t.Composite(e,n,i):new t(e,n,i)}static sanitizeNodeName(e){return e.replace(/\s/g,"_").replace(Hw,"")}static parseTrackName(e){let n=Yw.exec(e);if(n===null)throw new Error("THREE.PropertyBinding: Cannot parse trackName: "+e);let i={nodeName:n[2],objectName:n[3],objectIndex:n[4],propertyName:n[5],propertyIndex:n[6]},s=i.nodeName&&i.nodeName.lastIndexOf(".");if(s!==void 0&&s!==-1){let r=i.nodeName.substring(s+1);qw.indexOf(r)!==-1&&(i.nodeName=i.nodeName.substring(0,s),i.objectName=r)}if(i.propertyName===null||i.propertyName.length===0)throw new Error("THREE.PropertyBinding: can not parse propertyName from trackName: "+e);return i}static findNode(e,n){if(n===void 0||n===""||n==="."||n===-1||n===e.name||n===e.uuid)return e;if(e.skeleton){let i=e.skeleton.getBoneByName(n);if(i!==void 0)return i}if(e.children){let i=function(r){for(let a=0;a<r.length;a++){let o=r[a];if(o.name===n||o.uuid===n)return o;let l=i(o.children);if(l)return l}return null},s=i(e.children);if(s)return s}return null}_getValue_unavailable(){}_setValue_unavailable(){}_getValue_direct(e,n){e[n]=this.targetObject[this.propertyName]}_getValue_array(e,n){let i=this.resolvedProperty;for(let s=0,r=i.length;s!==r;++s)e[n++]=i[s]}_getValue_arrayElement(e,n){e[n]=this.resolvedProperty[this.propertyIndex]}_getValue_toArray(e,n){this.resolvedProperty.toArray(e,n)}_setValue_direct(e,n){this.targetObject[this.propertyName]=e[n]}_setValue_direct_setNeedsUpdate(e,n){this.targetObject[this.propertyName]=e[n],this.targetObject.needsUpdate=!0}_setValue_direct_setMatrixWorldNeedsUpdate(e,n){this.targetObject[this.propertyName]=e[n],this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_array(e,n){let i=this.resolvedProperty;for(let s=0,r=i.length;s!==r;++s)i[s]=e[n++]}_setValue_array_setNeedsUpdate(e,n){let i=this.resolvedProperty;for(let s=0,r=i.length;s!==r;++s)i[s]=e[n++];this.targetObject.needsUpdate=!0}_setValue_array_setMatrixWorldNeedsUpdate(e,n){let i=this.resolvedProperty;for(let s=0,r=i.length;s!==r;++s)i[s]=e[n++];this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_arrayElement(e,n){this.resolvedProperty[this.propertyIndex]=e[n]}_setValue_arrayElement_setNeedsUpdate(e,n){this.resolvedProperty[this.propertyIndex]=e[n],this.targetObject.needsUpdate=!0}_setValue_arrayElement_setMatrixWorldNeedsUpdate(e,n){this.resolvedProperty[this.propertyIndex]=e[n],this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_fromArray(e,n){this.resolvedProperty.fromArray(e,n)}_setValue_fromArray_setNeedsUpdate(e,n){this.resolvedProperty.fromArray(e,n),this.targetObject.needsUpdate=!0}_setValue_fromArray_setMatrixWorldNeedsUpdate(e,n){this.resolvedProperty.fromArray(e,n),this.targetObject.matrixWorldNeedsUpdate=!0}_getValue_unbound(e,n){this.bind(),this.getValue(e,n)}_setValue_unbound(e,n){this.bind(),this.setValue(e,n)}bind(){let e=this.node,n=this.parsedPath,i=n.objectName,s=n.propertyName,r=n.propertyIndex;if(e||(e=t.findNode(this.rootNode,n.nodeName),this.node=e),this.getValue=this._getValue_unavailable,this.setValue=this._setValue_unavailable,!e){De("PropertyBinding: No target node found for track: "+this.path+".");return}if(i){let c=n.objectIndex;switch(i){case"materials":if(!e.material){Ie("PropertyBinding: Can not bind to material as node does not have a material.",this);return}if(!e.material.materials){Ie("PropertyBinding: Can not bind to material.materials as node.material does not have a materials array.",this);return}e=e.material.materials;break;case"bones":if(!e.skeleton){Ie("PropertyBinding: Can not bind to bones as node does not have a skeleton.",this);return}e=e.skeleton.bones;for(let h=0;h<e.length;h++)if(e[h].name===c){c=h;break}break;case"map":if("map"in e){e=e.map;break}if(!e.material){Ie("PropertyBinding: Can not bind to material as node does not have a material.",this);return}if(!e.material.map){Ie("PropertyBinding: Can not bind to material.map as node.material does not have a map.",this);return}e=e.material.map;break;default:if(e[i]===void 0){Ie("PropertyBinding: Can not bind to objectName of node undefined.",this);return}e=e[i]}if(c!==void 0){if(e[c]===void 0){Ie("PropertyBinding: Trying to bind to objectIndex of objectName, but is undefined.",this,e);return}e=e[c]}}let a=e[s];if(a===void 0){let c=n.nodeName;Ie("PropertyBinding: Trying to update property for track: "+c+"."+s+" but it wasn't found.",e);return}let o=this.Versioning.None;this.targetObject=e,e.isMaterial===!0?o=this.Versioning.NeedsUpdate:e.isObject3D===!0&&(o=this.Versioning.MatrixWorldNeedsUpdate);let l=this.BindingType.Direct;if(r!==void 0){if(s==="morphTargetInfluences"){if(!e.geometry){Ie("PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.",this);return}if(!e.geometry.morphAttributes){Ie("PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.morphAttributes.",this);return}e.morphTargetDictionary[r]!==void 0&&(r=e.morphTargetDictionary[r])}l=this.BindingType.ArrayElement,this.resolvedProperty=a,this.propertyIndex=r}else a.fromArray!==void 0&&a.toArray!==void 0?(l=this.BindingType.HasFromToArray,this.resolvedProperty=a):Array.isArray(a)?(l=this.BindingType.EntireArray,this.resolvedProperty=a):this.propertyName=s;this.getValue=this.GetterByBindingType[l],this.setValue=this.SetterByBindingTypeAndVersioning[l][o]}unbind(){this.node=null,this.getValue=this._getValue_unbound,this.setValue=this._setValue_unbound}};Rt.Composite=n0;Rt.prototype.BindingType={Direct:0,EntireArray:1,ArrayElement:2,HasFromToArray:3};Rt.prototype.Versioning={None:0,NeedsUpdate:1,MatrixWorldNeedsUpdate:2};Rt.prototype.GetterByBindingType=[Rt.prototype._getValue_direct,Rt.prototype._getValue_array,Rt.prototype._getValue_arrayElement,Rt.prototype._getValue_toArray];Rt.prototype.SetterByBindingTypeAndVersioning=[[Rt.prototype._setValue_direct,Rt.prototype._setValue_direct_setNeedsUpdate,Rt.prototype._setValue_direct_setMatrixWorldNeedsUpdate],[Rt.prototype._setValue_array,Rt.prototype._setValue_array_setNeedsUpdate,Rt.prototype._setValue_array_setMatrixWorldNeedsUpdate],[Rt.prototype._setValue_arrayElement,Rt.prototype._setValue_arrayElement_setNeedsUpdate,Rt.prototype._setValue_arrayElement_setMatrixWorldNeedsUpdate],[Rt.prototype._setValue_fromArray,Rt.prototype._setValue_fromArray_setNeedsUpdate,Rt.prototype._setValue_fromArray_setMatrixWorldNeedsUpdate]];var J3=new Float32Array(1);var Nt=class t{constructor(e){this.value=e}clone(){return new t(this.value.clone===void 0?this.value:this.value.clone())}};var Mc=class{constructor(e=!0){this.autoStart=e,this.startTime=0,this.oldTime=0,this.elapsedTime=0,this.running=!1,De("Clock: This module has been deprecated. Please use THREE.Timer instead.")}start(){this.startTime=performance.now(),this.oldTime=this.startTime,this.elapsedTime=0,this.running=!0}stop(){this.getElapsedTime(),this.running=!1,this.autoStart=!1}getElapsedTime(){return this.getDelta(),this.elapsedTime}getDelta(){let e=0;if(this.autoStart&&!this.running)return this.start(),0;if(this.running){let n=performance.now();e=(n-this.oldTime)/1e3,this.oldTime=n,this.elapsedTime+=e}return e}};var L0=class L0{constructor(e,n,i,s){this.elements=[1,0,0,1],e!==void 0&&this.set(e,n,i,s)}identity(){return this.set(1,0,0,1),this}fromArray(e,n=0){for(let i=0;i<4;i++)this.elements[i]=e[i+n];return this}set(e,n,i,s){let r=this.elements;return r[0]=e,r[2]=n,r[1]=i,r[3]=s,this}};L0.prototype.isMatrix2=!0;var i0=L0;function U0(t,e,n,i){let s=Qw(i);switch(n){case M0:return t*e;case T0:return t*e/s.components*s.byteLength;case Bh:return t*e/s.components*s.byteLength;case Fr:return t*e*2/s.components*s.byteLength;case Nh:return t*e*2/s.components*s.byteLength;case E0:return t*e*3/s.components*s.byteLength;case Ri:return t*e*4/s.components*s.byteLength;case Ph:return t*e*4/s.components*s.byteLength;case wc:case Cc:return Math.floor((t+3)/4)*Math.floor((e+3)/4)*8;case Rc:case Dc:return Math.floor((t+3)/4)*Math.floor((e+3)/4)*16;case Oh:case zh:return Math.max(t,16)*Math.max(e,8)/4;case Lh:case Fh:return Math.max(t,8)*Math.max(e,8)/2;case Hh:case Gh:case kh:case Wh:return Math.floor((t+3)/4)*Math.floor((e+3)/4)*8;case Vh:case Uc:case Xh:return Math.floor((t+3)/4)*Math.floor((e+3)/4)*16;case Yh:return Math.floor((t+3)/4)*Math.floor((e+3)/4)*16;case qh:return Math.floor((t+4)/5)*Math.floor((e+3)/4)*16;case Qh:return Math.floor((t+4)/5)*Math.floor((e+4)/5)*16;case Zh:return Math.floor((t+5)/6)*Math.floor((e+4)/5)*16;case Kh:return Math.floor((t+5)/6)*Math.floor((e+5)/6)*16;case Jh:return Math.floor((t+7)/8)*Math.floor((e+4)/5)*16;case jh:return Math.floor((t+7)/8)*Math.floor((e+5)/6)*16;case $h:return Math.floor((t+7)/8)*Math.floor((e+7)/8)*16;case ed:return Math.floor((t+9)/10)*Math.floor((e+4)/5)*16;case td:return Math.floor((t+9)/10)*Math.floor((e+5)/6)*16;case nd:return Math.floor((t+9)/10)*Math.floor((e+7)/8)*16;case id:return Math.floor((t+9)/10)*Math.floor((e+9)/10)*16;case sd:return Math.floor((t+11)/12)*Math.floor((e+9)/10)*16;case rd:return Math.floor((t+11)/12)*Math.floor((e+11)/12)*16;case ad:case od:case ld:return Math.ceil(t/4)*Math.ceil(e/4)*16;case cd:case ud:return Math.ceil(t/4)*Math.ceil(e/4)*8;case Ic:case fd:return Math.ceil(t/4)*Math.ceil(e/4)*16}throw new Error(`Unable to determine texture byte length for ${n} format.`)}function Qw(t){switch(t){case Xt:case y0:return{byteLength:1,components:1};case Ko:case _0:case Xi:return{byteLength:2,components:1};case Uh:case Ih:return{byteLength:2,components:4};case Wi:case Dh:case fi:return{byteLength:4,components:1};case S0:case A0:return{byteLength:4,components:3}}throw new Error(`THREE.TextureUtils: Unknown texture type ${t}.`)}typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register",{detail:{revision:"186"}}));typeof window<"u"&&(window.__THREE__?De("WARNING: Multiple instances of Three.js being imported."):window.__THREE__="186");function uM(){let t=null,e=!1,n=null,i=null;function s(r,a){i=t.requestAnimationFrame(s),n(r,a)}return{start:function(){e!==!0&&n!==null&&t!==null&&(i=t.requestAnimationFrame(s),e=!0)},stop:function(){t!==null&&t.cancelAnimationFrame(i),e=!1},setAnimationLoop:function(r){n=r},setContext:function(r){t=r}}}function Zw(t){let e=new WeakMap;function n(o,l){let c=o.array,h=o.usage,p=c.byteLength,u=t.createBuffer();t.bindBuffer(l,u),t.bufferData(l,c,h),o.onUploadCallback();let d;if(c instanceof Float32Array)d=t.FLOAT;else if(typeof Float16Array<"u"&&c instanceof Float16Array)d=t.HALF_FLOAT;else if(c instanceof Uint16Array)o.isFloat16BufferAttribute?d=t.HALF_FLOAT:d=t.UNSIGNED_SHORT;else if(c instanceof Int16Array)d=t.SHORT;else if(c instanceof Uint32Array)d=t.UNSIGNED_INT;else if(c instanceof Int32Array)d=t.INT;else if(c instanceof Int8Array)d=t.BYTE;else if(c instanceof Uint8Array)d=t.UNSIGNED_BYTE;else if(c instanceof Uint8ClampedArray)d=t.UNSIGNED_BYTE;else throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: "+c);return{buffer:u,type:d,bytesPerElement:c.BYTES_PER_ELEMENT,version:o.version,size:p}}function i(o,l,c){let h=l.array,p=l.updateRanges;if(t.bindBuffer(c,o),p.length===0)t.bufferSubData(c,0,h);else{p.sort((d,v)=>d.start-v.start);let u=0;for(let d=1;d<p.length;d++){let v=p[u],M=p[d];M.start<=v.start+v.count+1?v.count=Math.max(v.count,M.start+M.count-v.start):(++u,p[u]=M)}p.length=u+1;for(let d=0,v=p.length;d<v;d++){let M=p[d];t.bufferSubData(c,M.start*h.BYTES_PER_ELEMENT,h,M.start,M.count)}l.clearUpdateRanges()}l.onUploadCallback()}function s(o){return o.isInterleavedBufferAttribute&&(o=o.data),e.get(o)}function r(o){o.isInterleavedBufferAttribute&&(o=o.data);let l=e.get(o);l&&(t.deleteBuffer(l.buffer),e.delete(o))}function a(o,l){if(o.isInterleavedBufferAttribute&&(o=o.data),o.isGLBufferAttribute){let h=e.get(o);(!h||h.version<o.version)&&e.set(o,{buffer:o.buffer,type:o.type,bytesPerElement:o.elementSize,version:o.version});return}let c=e.get(o);if(c===void 0)e.set(o,n(o,l));else if(c.version<o.version){if(c.size!==o.array.byteLength)throw new Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");i(c.buffer,o,l),c.version=o.version}}return{get:s,remove:r,update:a}}var Kw=`#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`,Jw=`#ifdef USE_ALPHAHASH
	const float ALPHA_HASH_SCALE = 0.05;
	float hash2D( vec2 value ) {
		return fract( 1.0e4 * sin( 17.0 * value.x + 0.1 * value.y ) * ( 0.1 + abs( sin( 13.0 * value.y + value.x ) ) ) );
	}
	float hash3D( vec3 value ) {
		return hash2D( vec2( hash2D( value.xy ), value.z ) );
	}
	float getAlphaHashThreshold( vec3 position ) {
		float maxDeriv = max(
			length( dFdx( position.xyz ) ),
			length( dFdy( position.xyz ) )
		);
		float pixScale = 1.0 / ( ALPHA_HASH_SCALE * maxDeriv );
		vec2 pixScales = vec2(
			exp2( floor( log2( pixScale ) ) ),
			exp2( ceil( log2( pixScale ) ) )
		);
		vec2 alpha = vec2(
			hash3D( floor( pixScales.x * position.xyz ) ),
			hash3D( floor( pixScales.y * position.xyz ) )
		);
		float lerpFactor = fract( log2( pixScale ) );
		float x = ( 1.0 - lerpFactor ) * alpha.x + lerpFactor * alpha.y;
		float a = min( lerpFactor, 1.0 - lerpFactor );
		vec3 cases = vec3(
			x * x / ( 2.0 * a * ( 1.0 - a ) ),
			( x - 0.5 * a ) / ( 1.0 - a ),
			1.0 - ( ( 1.0 - x ) * ( 1.0 - x ) / ( 2.0 * a * ( 1.0 - a ) ) )
		);
		float threshold = ( x < ( 1.0 - a ) )
			? ( ( x < a ) ? cases.x : cases.y )
			: cases.z;
		return clamp( threshold , 1.0e-6, 1.0 );
	}
#endif`,jw=`#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`,$w=`#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,eC=`#ifdef USE_ALPHATEST
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < alphaTest ) discard;
	#endif
#endif`,tC=`#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`,nC=`#ifdef USE_AOMAP
	float ambientOcclusion = ( texture2D( aoMap, vAoMapUv ).r - 1.0 ) * aoMapIntensity + 1.0;
	reflectedLight.indirectDiffuse *= ambientOcclusion;
	#if defined( USE_CLEARCOAT ) 
		clearcoatSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_SHEEN ) 
		sheenSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD )
		float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
		reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
	#endif
#endif`,iC=`#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`,sC=`#ifdef USE_BATCHING
	#if ! defined( GL_ANGLE_multi_draw )
	#define gl_DrawID _gl_DrawID
	uniform int _gl_DrawID;
	#endif
	uniform highp sampler2D batchingTexture;
	uniform highp usampler2D batchingIdTexture;
	mat4 getBatchingMatrix( const in float i ) {
		int size = textureSize( batchingTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( batchingTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( batchingTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( batchingTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( batchingTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
	float getIndirectIndex( const in int i ) {
		int size = textureSize( batchingIdTexture, 0 ).x;
		int x = i % size;
		int y = i / size;
		return float( texelFetch( batchingIdTexture, ivec2( x, y ), 0 ).r );
	}
#endif
#ifdef USE_BATCHING_COLOR
	uniform sampler2D batchingColorTexture;
	vec4 getBatchingColor( const in float i ) {
		int size = textureSize( batchingColorTexture, 0 ).x;
		int j = int( i );
		int x = j % size;
		int y = j / size;
		return texelFetch( batchingColorTexture, ivec2( x, y ), 0 );
	}
#endif`,rC=`#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );
#endif`,aC=`vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`,oC=`vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`,lC=`float G_BlinnPhong_Implicit( ) {
	return 0.25;
}
float D_BlinnPhong( const in float shininess, const in float dotNH ) {
	return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );
}
vec3 BRDF_BlinnPhong( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float shininess ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( specularColor, 1.0, dotVH );
	float G = G_BlinnPhong_Implicit( );
	float D = D_BlinnPhong( shininess, dotNH );
	return F * ( G * D );
} // validated`,cC=`#ifdef USE_IRIDESCENCE
	const mat3 XYZ_TO_REC709 = mat3(
		 3.2404542, -0.9692660,  0.0556434,
		-1.5371385,  1.8760108, -0.2040259,
		-0.4985314,  0.0415560,  1.0572252
	);
	vec3 Fresnel0ToIor( vec3 fresnel0 ) {
		vec3 sqrtF0 = sqrt( fresnel0 );
		return ( vec3( 1.0 ) + sqrtF0 ) / ( vec3( 1.0 ) - sqrtF0 );
	}
	vec3 IorToFresnel0( vec3 transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - vec3( incidentIor ) ) / ( transmittedIor + vec3( incidentIor ) ) );
	}
	float IorToFresnel0( float transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - incidentIor ) / ( transmittedIor + incidentIor ));
	}
	vec3 evalSensitivity( float OPD, vec3 shift ) {
		float phase = 2.0 * PI * OPD * 1.0e-9;
		vec3 val = vec3( 5.4856e-13, 4.4201e-13, 5.2481e-13 );
		vec3 pos = vec3( 1.6810e+06, 1.7953e+06, 2.2084e+06 );
		vec3 var = vec3( 4.3278e+09, 9.3046e+09, 6.6121e+09 );
		vec3 xyz = val * sqrt( 2.0 * PI * var ) * cos( pos * phase + shift ) * exp( - pow2( phase ) * var );
		xyz.x += 9.7470e-14 * sqrt( 2.0 * PI * 4.5282e+09 ) * cos( 2.2399e+06 * phase + shift[ 0 ] ) * exp( - 4.5282e+09 * pow2( phase ) );
		xyz /= 1.0685e-7;
		vec3 rgb = XYZ_TO_REC709 * xyz;
		return rgb;
	}
	vec3 evalIridescence( float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0 ) {
		vec3 I;
		float iridescenceIOR = mix( outsideIOR, eta2, smoothstep( 0.0, 0.03, thinFilmThickness ) );
		float sinTheta2Sq = pow2( outsideIOR / iridescenceIOR ) * ( 1.0 - pow2( cosTheta1 ) );
		float cosTheta2Sq = 1.0 - sinTheta2Sq;
		if ( cosTheta2Sq < 0.0 ) {
			return vec3( 1.0 );
		}
		float cosTheta2 = sqrt( cosTheta2Sq );
		float R0 = IorToFresnel0( iridescenceIOR, outsideIOR );
		float R12 = F_Schlick( R0, 1.0, cosTheta1 );
		float T121 = 1.0 - R12;
		float phi12 = 0.0;
		if ( iridescenceIOR < outsideIOR ) phi12 = PI;
		float phi21 = PI - phi12;
		vec3 baseIOR = Fresnel0ToIor( clamp( baseF0, 0.0, 0.9999 ) );		vec3 R1 = IorToFresnel0( baseIOR, iridescenceIOR );
		vec3 R23 = F_Schlick( R1, 1.0, cosTheta2 );
		vec3 phi23 = vec3( 0.0 );
		if ( baseIOR[ 0 ] < iridescenceIOR ) phi23[ 0 ] = PI;
		if ( baseIOR[ 1 ] < iridescenceIOR ) phi23[ 1 ] = PI;
		if ( baseIOR[ 2 ] < iridescenceIOR ) phi23[ 2 ] = PI;
		float OPD = 2.0 * iridescenceIOR * thinFilmThickness * cosTheta2;
		vec3 phi = vec3( phi21 ) + phi23;
		vec3 R123 = clamp( R12 * R23, 1e-5, 0.9999 );
		vec3 r123 = sqrt( R123 );
		vec3 Rs = pow2( T121 ) * R23 / ( vec3( 1.0 ) - R123 );
		vec3 C0 = R12 + Rs;
		I = C0;
		vec3 Cm = Rs - T121;
		for ( int m = 1; m <= 2; ++ m ) {
			Cm *= r123;
			vec3 Sm = 2.0 * evalSensitivity( float( m ) * OPD, float( m ) * phi );
			I += Cm * Sm;
		}
		return max( I, vec3( 0.0 ) );
	}
#endif`,uC=`#ifdef USE_BUMPMAP
	uniform sampler2D bumpMap;
	uniform float bumpScale;
	vec2 dHdxy_fwd() {
		vec2 dSTdx = dFdx( vBumpMapUv );
		vec2 dSTdy = dFdy( vBumpMapUv );
		float Hll = bumpScale * texture2D( bumpMap, vBumpMapUv ).x;
		float dBx = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdx ).x - Hll;
		float dBy = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdy ).x - Hll;
		return vec2( dBx, dBy );
	}
	vec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDirection ) {
		vec3 vSigmaX = normalize( dFdx( surf_pos.xyz ) );
		vec3 vSigmaY = normalize( dFdy( surf_pos.xyz ) );
		vec3 vN = surf_norm;
		vec3 R1 = cross( vSigmaY, vN );
		vec3 R2 = cross( vN, vSigmaX );
		float fDet = dot( vSigmaX, R1 ) * faceDirection;
		vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );
		return normalize( abs( fDet ) * surf_norm - vGrad );
	}
#endif`,fC=`#if NUM_CLIPPING_PLANES > 0
	vec4 plane;
	#ifdef ALPHA_TO_COVERAGE
		float distanceToPlane, distanceGradient;
		float clipOpacity = 1.0;
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
			distanceGradient = fwidth( distanceToPlane ) / 2.0;
			clipOpacity *= smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			if ( clipOpacity == 0.0 ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			float unionClipOpacity = 1.0;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
				distanceGradient = fwidth( distanceToPlane ) / 2.0;
				unionClipOpacity *= 1.0 - smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			}
			#pragma unroll_loop_end
			clipOpacity *= 1.0 - unionClipOpacity;
		#endif
		diffuseColor.a *= clipOpacity;
		if ( diffuseColor.a == 0.0 ) discard;
	#else
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			if ( dot( vClipPosition, plane.xyz ) > plane.w ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			bool clipped = true;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				clipped = ( dot( vClipPosition, plane.xyz ) > plane.w ) && clipped;
			}
			#pragma unroll_loop_end
			if ( clipped ) discard;
		#endif
	#endif
#endif`,hC=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`,dC=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`,pC=`#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`,mC=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#endif`,gC=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#endif`,vC=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	varying vec4 vColor;
#endif`,xC=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	vColor = vec4( 1.0 );
#endif
#ifdef USE_COLOR_ALPHA
	vColor *= color;
#elif defined( USE_COLOR )
	vColor.rgb *= color;
#endif
#ifdef USE_INSTANCING_COLOR
	vColor.rgb *= instanceColor.rgb;
#endif
#ifdef USE_BATCHING_COLOR
	vColor *= getBatchingColor( getIndirectIndex( gl_DrawID ) );
#endif`,yC=`#define PI 3.141592653589793
#define PI2 6.283185307179586
#define PI_HALF 1.5707963267948966
#define RECIPROCAL_PI 0.3183098861837907
#define RECIPROCAL_PI2 0.15915494309189535
#define EPSILON 1e-6
#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
#define whiteComplement( a ) ( 1.0 - saturate( a ) )
float pow2( const in float x ) { return x*x; }
vec3 pow2( const in vec3 x ) { return x*x; }
float pow3( const in float x ) { return x*x*x; }
float pow4( const in float x ) { float x2 = x*x; return x2*x2; }
float max3( const in vec3 v ) { return max( max( v.x, v.y ), v.z ); }
float average( const in vec3 v ) { return dot( v, vec3( 0.3333333 ) ); }
highp float rand( const in vec2 uv ) {
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract( sin( sn ) * c );
}
#ifdef HIGH_PRECISION
	float precisionSafeLength( vec3 v ) { return length( v ); }
#else
	float precisionSafeLength( vec3 v ) {
		float maxComponent = max3( abs( v ) );
		return length( v / maxComponent ) * maxComponent;
	}
#endif
struct IncidentLight {
	vec3 color;
	vec3 direction;
	bool visible;
};
struct ReflectedLight {
	vec3 directDiffuse;
	vec3 directSpecular;
	vec3 indirectDiffuse;
	vec3 indirectSpecular;
};
#ifdef USE_ALPHAHASH
	varying vec3 vPosition;
#endif
vec3 transformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );
}
#define inverseTransformDirection transformDirectionByInverseViewMatrix
vec3 transformNormalByInverseViewMatrix( in vec3 normal, in mat4 viewMatrix ) {
	return normalize( ( vec4( normal, 0.0 ) * viewMatrix ).xyz );
}
vec3 transformDirectionByInverseViewMatrix( in vec3 dir, in mat4 viewMatrix ) {
	return normalize( ( vec4( dir, 0.0 ) * viewMatrix ).xyz );
}
bool isPerspectiveMatrix( mat4 m ) {
	return m[ 2 ][ 3 ] == - 1.0;
}
vec2 equirectUv( in vec3 dir ) {
	float u = atan( dir.z, dir.x ) * RECIPROCAL_PI2 + 0.5;
	float v = asin( clamp( dir.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;
	return vec2( u, v );
}
vec3 BRDF_Lambert( const in vec3 diffuseColor ) {
	return RECIPROCAL_PI * diffuseColor;
}
vec3 F_Schlick( const in vec3 f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
}
float F_Schlick( const in float f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
} // validated`,_C=`#ifdef ENVMAP_TYPE_CUBE_UV
	#define cubeUV_minMipLevel 4.0
	#define cubeUV_minTileSize 16.0
	float getFace( vec3 direction ) {
		vec3 absDirection = abs( direction );
		float face = - 1.0;
		if ( absDirection.x > absDirection.z ) {
			if ( absDirection.x > absDirection.y )
				face = direction.x > 0.0 ? 0.0 : 3.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		} else {
			if ( absDirection.z > absDirection.y )
				face = direction.z > 0.0 ? 2.0 : 5.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		}
		return face;
	}
	vec2 getUV( vec3 direction, float face ) {
		vec2 uv;
		if ( face == 0.0 ) {
			uv = vec2( direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 1.0 ) {
			uv = vec2( - direction.x, - direction.z ) / abs( direction.y );
		} else if ( face == 2.0 ) {
			uv = vec2( - direction.x, direction.y ) / abs( direction.z );
		} else if ( face == 3.0 ) {
			uv = vec2( - direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 4.0 ) {
			uv = vec2( - direction.x, direction.z ) / abs( direction.y );
		} else {
			uv = vec2( direction.x, direction.y ) / abs( direction.z );
		}
		return 0.5 * ( uv + 1.0 );
	}
	vec3 bilinearCubeUV( sampler2D envMap, vec3 direction, float mipInt ) {
		float face = getFace( direction );
		float filterInt = max( cubeUV_minMipLevel - mipInt, 0.0 );
		mipInt = max( mipInt, cubeUV_minMipLevel );
		float faceSize = exp2( mipInt );
		highp vec2 uv = getUV( direction, face ) * ( faceSize - 2.0 ) + 1.0;
		if ( face > 2.0 ) {
			uv.y += faceSize;
			face -= 3.0;
		}
		uv.x += face * faceSize;
		uv.x += filterInt * 3.0 * cubeUV_minTileSize;
		uv.y += 4.0 * ( exp2( CUBEUV_MAX_MIP ) - faceSize );
		uv.x *= CUBEUV_TEXEL_WIDTH;
		uv.y *= CUBEUV_TEXEL_HEIGHT;
		#ifdef texture2DGradEXT
			return texture2DGradEXT( envMap, uv, vec2( 0.0 ), vec2( 0.0 ) ).rgb;
		#else
			return texture2D( envMap, uv ).rgb;
		#endif
	}
	#define cubeUV_r0 1.0
	#define cubeUV_m0 - 2.0
	#define cubeUV_r1 0.8
	#define cubeUV_m1 - 1.0
	#define cubeUV_r4 0.4
	#define cubeUV_m4 2.0
	#define cubeUV_r5 0.305
	#define cubeUV_m5 3.0
	#define cubeUV_r6 0.21
	#define cubeUV_m6 4.0
	float roughnessToMip( float roughness ) {
		float mip = 0.0;
		if ( roughness >= cubeUV_r1 ) {
			mip = ( cubeUV_r0 - roughness ) * ( cubeUV_m1 - cubeUV_m0 ) / ( cubeUV_r0 - cubeUV_r1 ) + cubeUV_m0;
		} else if ( roughness >= cubeUV_r4 ) {
			mip = ( cubeUV_r1 - roughness ) * ( cubeUV_m4 - cubeUV_m1 ) / ( cubeUV_r1 - cubeUV_r4 ) + cubeUV_m1;
		} else if ( roughness >= cubeUV_r5 ) {
			mip = ( cubeUV_r4 - roughness ) * ( cubeUV_m5 - cubeUV_m4 ) / ( cubeUV_r4 - cubeUV_r5 ) + cubeUV_m4;
		} else if ( roughness >= cubeUV_r6 ) {
			mip = ( cubeUV_r5 - roughness ) * ( cubeUV_m6 - cubeUV_m5 ) / ( cubeUV_r5 - cubeUV_r6 ) + cubeUV_m5;
		} else {
			mip = - 2.0 * log2( 1.16 * roughness );		}
		return mip;
	}
	vec4 textureCubeUV( sampler2D envMap, vec3 sampleDir, float roughness ) {
		float mip = clamp( roughnessToMip( roughness ), cubeUV_m0, CUBEUV_MAX_MIP );
		float mipF = fract( mip );
		float mipInt = floor( mip );
		vec3 color0 = bilinearCubeUV( envMap, sampleDir, mipInt );
		if ( mipF == 0.0 ) {
			return vec4( color0, 1.0 );
		} else {
			vec3 color1 = bilinearCubeUV( envMap, sampleDir, mipInt + 1.0 );
			return vec4( mix( color0, color1, mipF ), 1.0 );
		}
	}
#endif`,SC=`vec3 transformedNormal = objectNormal;
#ifdef USE_TANGENT
	vec3 transformedTangent = objectTangent;
#endif
#ifdef USE_BATCHING
	mat3 bm = mat3( batchingMatrix );
	transformedNormal /= vec3( dot( bm[ 0 ], bm[ 0 ] ), dot( bm[ 1 ], bm[ 1 ] ), dot( bm[ 2 ], bm[ 2 ] ) );
	transformedNormal = bm * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = bm * transformedTangent;
	#endif
#endif
#ifdef USE_INSTANCING
	mat3 im = mat3( instanceMatrix );
	transformedNormal /= vec3( dot( im[ 0 ], im[ 0 ] ), dot( im[ 1 ], im[ 1 ] ), dot( im[ 2 ], im[ 2 ] ) );
	transformedNormal = im * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = im * transformedTangent;
	#endif
#endif
transformedNormal = normalMatrix * transformedNormal;
#ifdef FLIP_SIDED
	transformedNormal = - transformedNormal;
#endif
#ifdef USE_TANGENT
	transformedTangent = ( modelViewMatrix * vec4( transformedTangent, 0.0 ) ).xyz;
#endif`,AC=`#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`,MC=`#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`,EC=`#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE
		emissiveColor = sRGBTransferEOTF( emissiveColor );
	#endif
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`,TC=`#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`,bC="gl_FragColor = linearToOutputTexel( gl_FragColor );",wC=`vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferEOTF( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`,CC=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vec3 cameraToFrag;
		if ( isOrthographic ) {
			cameraToFrag = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToFrag = normalize( vWorldPosition - cameraPosition );
		}
		vec3 worldNormal = transformNormalByInverseViewMatrix( normal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vec3 reflectVec = reflect( cameraToFrag, worldNormal );
		#else
			vec3 reflectVec = refract( cameraToFrag, worldNormal, refractionRatio );
		#endif
	#else
		vec3 reflectVec = vReflect;
	#endif
	#ifdef ENVMAP_TYPE_CUBE
		vec4 envColor = textureCube( envMap, envMapRotation * reflectVec );
		#ifdef ENVMAP_BLENDING_MULTIPLY
			outgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );
		#elif defined( ENVMAP_BLENDING_MIX )
			outgoingLight = mix( outgoingLight, envColor.xyz, specularStrength * reflectivity );
		#elif defined( ENVMAP_BLENDING_ADD )
			outgoingLight += envColor.xyz * specularStrength * reflectivity;
		#endif
	#endif
#endif`,RC=`#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform mat3 envMapRotation;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif
#endif`,DC=`#ifdef USE_ENVMAP
	uniform float reflectivity;
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		varying vec3 vWorldPosition;
		uniform float refractionRatio;
	#else
		varying vec3 vReflect;
	#endif
#endif`,UC=`#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		
		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`,IC=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vWorldPosition = worldPosition.xyz;
	#else
		vec3 cameraToVertex;
		if ( isOrthographic ) {
			cameraToVertex = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToVertex = normalize( worldPosition.xyz - cameraPosition );
		}
		vec3 worldNormal = transformNormalByInverseViewMatrix( transformedNormal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vReflect = reflect( cameraToVertex, worldNormal );
		#else
			vReflect = refract( cameraToVertex, worldNormal, refractionRatio );
		#endif
	#endif
#endif`,BC=`#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`,NC=`#ifdef USE_FOG
	varying float vFogDepth;
#endif`,PC=`#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`,LC=`#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`,OC=`#ifdef USE_GRADIENTMAP
	uniform sampler2D gradientMap;
#endif
vec3 getGradientIrradiance( vec3 normal, vec3 lightDirection ) {
	float dotNL = dot( normal, lightDirection );
	vec2 coord = vec2( dotNL * 0.5 + 0.5, 0.0 );
	#ifdef USE_GRADIENTMAP
		return vec3( texture2D( gradientMap, coord ).r );
	#else
		vec2 fw = fwidth( coord ) * 0.5;
		return mix( vec3( 0.7 ), vec3( 1.0 ), smoothstep( 0.7 - fw.x, 0.7 + fw.x, coord.x ) );
	#endif
}`,FC=`#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`,zC=`LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`,HC=`varying vec3 vViewPosition;
struct LambertMaterial {
	vec3 diffuseColor;
	float specularStrength;
};
void RE_Direct_Lambert( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Lambert( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Lambert
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`,GC=`uniform bool receiveShadow;
uniform vec3 ambientLightColor;
#if defined( USE_LIGHT_PROBES )
	uniform vec3 lightProbe[ 9 ];
#endif
vec3 shGetIrradianceAt( in vec3 normal, in vec3 shCoefficients[ 9 ] ) {
	float x = normal.x, y = normal.y, z = normal.z;
	vec3 result = shCoefficients[ 0 ] * 0.886227;
	result += shCoefficients[ 1 ] * 2.0 * 0.511664 * y;
	result += shCoefficients[ 2 ] * 2.0 * 0.511664 * z;
	result += shCoefficients[ 3 ] * 2.0 * 0.511664 * x;
	result += shCoefficients[ 4 ] * 2.0 * 0.429043 * x * y;
	result += shCoefficients[ 5 ] * 2.0 * 0.429043 * y * z;
	result += shCoefficients[ 6 ] * ( 0.743125 * z * z - 0.247708 );
	result += shCoefficients[ 7 ] * 2.0 * 0.429043 * x * z;
	result += shCoefficients[ 8 ] * 0.429043 * ( x * x - y * y );
	return result;
}
vec3 getLightProbeIrradiance( const in vec3 lightProbe[ 9 ], const in vec3 normal ) {
	vec3 worldNormal = transformNormalByInverseViewMatrix( normal, viewMatrix );
	vec3 irradiance = shGetIrradianceAt( worldNormal, lightProbe );
	return irradiance;
}
vec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {
	vec3 irradiance = ambientLightColor;
	return irradiance;
}
float getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {
	float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );
	if ( cutoffDistance > 0.0 ) {
		distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );
	}
	return distanceFalloff;
}
float getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {
	return smoothstep( coneCosine, penumbraCosine, angleCosine );
}
#if NUM_SUN_LIGHTS > 0
	struct SunLight {
		vec3 direction;
		vec3 color;
	};
	uniform SunLight sunLights[ NUM_SUN_LIGHTS ];
	void getSunLightInfo( const in SunLight sunLight, out IncidentLight light ) {
		light.color = sunLight.color;
		light.direction = sunLight.direction;
		light.visible = true;
	}
#endif
#if NUM_DIR_LIGHTS > 0
	struct DirectionalLight {
		vec3 direction;
		vec3 color;
	};
	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
	void getDirectionalLightInfo( const in DirectionalLight directionalLight, out IncidentLight light ) {
		light.color = directionalLight.color;
		light.direction = directionalLight.direction;
		light.visible = true;
	}
#endif
#if NUM_POINT_LIGHTS > 0
	struct PointLight {
		vec3 position;
		vec3 color;
		float distance;
		float decay;
	};
	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
	void getPointLightInfo( const in PointLight pointLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = pointLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float lightDistance = length( lVector );
		light.color = pointLight.color;
		light.color *= getDistanceAttenuation( lightDistance, pointLight.distance, pointLight.decay );
		light.visible = ( light.color != vec3( 0.0 ) );
	}
#endif
#if NUM_SPOT_LIGHTS > 0
	struct SpotLight {
		vec3 position;
		vec3 direction;
		vec3 color;
		float distance;
		float decay;
		float coneCos;
		float penumbraCos;
	};
	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];
	void getSpotLightInfo( const in SpotLight spotLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = spotLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float angleCos = dot( light.direction, spotLight.direction );
		float spotAttenuation = getSpotAttenuation( spotLight.coneCos, spotLight.penumbraCos, angleCos );
		if ( spotAttenuation > 0.0 ) {
			float lightDistance = length( lVector );
			light.color = spotLight.color * spotAttenuation;
			light.color *= getDistanceAttenuation( lightDistance, spotLight.distance, spotLight.decay );
			light.visible = ( light.color != vec3( 0.0 ) );
		} else {
			light.color = vec3( 0.0 );
			light.visible = false;
		}
	}
#endif
#if NUM_RECT_AREA_LIGHTS > 0
	struct RectAreaLight {
		vec3 color;
		vec3 position;
		vec3 halfWidth;
		vec3 halfHeight;
	};
	uniform sampler2D ltc_1;	uniform sampler2D ltc_2;
	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];
#endif
#if NUM_HEMI_LIGHTS > 0
	struct HemisphereLight {
		vec3 direction;
		vec3 skyColor;
		vec3 groundColor;
	};
	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];
	vec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in vec3 normal ) {
		float dotNL = dot( normal, hemiLight.direction );
		float hemiDiffuseWeight = 0.5 * dotNL + 0.5;
		vec3 irradiance = mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );
		return irradiance;
	}
#endif
#include <lightprobes_pars_fragment>`,VC=`#ifdef USE_ENVMAP
	vec3 getIBLIrradiance( const in vec3 normal ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 worldNormal = transformNormalByInverseViewMatrix( normal, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * worldNormal, 1.0 );
			return PI * envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	vec3 getIBLRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 reflectVec = reflect( - viewDir, normal );
			reflectVec = normalize( mix( reflectVec, normal, pow4( roughness ) ) );
			reflectVec = transformDirectionByInverseViewMatrix( reflectVec, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * reflectVec, roughness );
			return envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	#ifdef USE_RETROREFLECTION
		vec3 getIBLRetroRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {
			#ifdef ENVMAP_TYPE_CUBE_UV
				vec3 retroVec = normalize( mix( viewDir, normal, pow4( roughness ) ) );
				retroVec = transformDirectionByInverseViewMatrix( retroVec, viewMatrix );
				vec4 envMapColor = textureCubeUV( envMap, envMapRotation * retroVec, roughness );
				return envMapColor.rgb * envMapIntensity;
			#else
				return vec3( 0.0 );
			#endif
		}
	#endif
	#ifdef USE_ANISOTROPY
		vec3 getIBLAnisotropyRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {
			#ifdef ENVMAP_TYPE_CUBE_UV
				vec3 bentNormal = cross( bitangent, viewDir );
				bentNormal = normalize( cross( bentNormal, bitangent ) );
				bentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );
				return getIBLRadiance( viewDir, bentNormal, roughness );
			#else
				return vec3( 0.0 );
			#endif
		}
		#ifdef USE_RETROREFLECTION
			vec3 getIBLAnisotropyRetroRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {
				#ifdef ENVMAP_TYPE_CUBE_UV
					vec3 bentNormal = cross( bitangent, viewDir );
					bentNormal = normalize( cross( bentNormal, bitangent ) );
					bentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );
					return getIBLRetroRadiance( viewDir, bentNormal, roughness );
				#else
					return vec3( 0.0 );
				#endif
			}
		#endif
	#endif
#endif`,kC=`ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`,WC=`varying vec3 vViewPosition;
struct ToonMaterial {
	vec3 diffuseColor;
};
void RE_Direct_Toon( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 irradiance = getGradientIrradiance( geometryNormal, directLight.direction ) * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Toon( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Toon
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`,XC=`BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`,YC=`varying vec3 vViewPosition;
struct BlinnPhongMaterial {
	vec3 diffuseColor;
	vec3 specularColor;
	float specularShininess;
	float specularStrength;
};
void RE_Direct_BlinnPhong( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
	reflectedLight.directSpecular += irradiance * BRDF_BlinnPhong( directLight.direction, geometryViewDir, geometryNormal, material.specularColor, material.specularShininess ) * material.specularStrength;
}
void RE_IndirectDiffuse_BlinnPhong( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_BlinnPhong
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`,qC=`PhysicalMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.diffuseContribution = diffuseColor.rgb * ( 1.0 - metalnessFactor );
material.metalness = metalnessFactor;
vec3 dxy = max( abs( dFdx( nonPerturbedNormal ) ), abs( dFdy( nonPerturbedNormal ) ) );
float geometryRoughness = max( max( dxy.x, dxy.y ), dxy.z );
material.roughness = max( roughnessFactor, 0.0525 );material.roughness += geometryRoughness;
material.roughness = min( material.roughness, 1.0 );
#ifdef IOR
	material.ior = ior;
	#ifdef USE_SPECULAR
		float specularIntensityFactor = specularIntensity;
		vec3 specularColorFactor = specularColor;
		#ifdef USE_SPECULAR_COLORMAP
			specularColorFactor *= texture2D( specularColorMap, vSpecularColorMapUv ).rgb;
		#endif
		#ifdef USE_SPECULAR_INTENSITYMAP
			specularIntensityFactor *= texture2D( specularIntensityMap, vSpecularIntensityMapUv ).a;
		#endif
		material.specularF90 = mix( specularIntensityFactor, 1.0, metalnessFactor );
	#else
		float specularIntensityFactor = 1.0;
		vec3 specularColorFactor = vec3( 1.0 );
		material.specularF90 = 1.0;
	#endif
	material.specularColor = min( pow2( ( material.ior - 1.0 ) / ( material.ior + 1.0 ) ) * specularColorFactor, vec3( 1.0 ) ) * specularIntensityFactor;
	material.specularColorBlended = mix( material.specularColor, diffuseColor.rgb, metalnessFactor );
#else
	material.specularColor = vec3( 0.04 );
	material.specularColorBlended = mix( material.specularColor, diffuseColor.rgb, metalnessFactor );
	material.specularF90 = 1.0;
#endif
#ifdef USE_CLEARCOAT
	material.clearcoat = clearcoat;
	material.clearcoatRoughness = clearcoatRoughness;
	material.clearcoatF0 = vec3( 0.04 );
	material.clearcoatF90 = 1.0;
	#ifdef USE_CLEARCOATMAP
		material.clearcoat *= texture2D( clearcoatMap, vClearcoatMapUv ).x;
	#endif
	#ifdef USE_CLEARCOAT_ROUGHNESSMAP
		material.clearcoatRoughness *= texture2D( clearcoatRoughnessMap, vClearcoatRoughnessMapUv ).y;
	#endif
	material.clearcoat = saturate( material.clearcoat );	material.clearcoatRoughness = max( material.clearcoatRoughness, 0.0525 );
	material.clearcoatRoughness += geometryRoughness;
	material.clearcoatRoughness = min( material.clearcoatRoughness, 1.0 );
#endif
#ifdef USE_DISPERSION
	material.dispersion = dispersion;
#endif
#ifdef USE_RETROREFLECTION
	material.retroreflectivity = retroreflectivity;
#endif
#ifdef USE_IRIDESCENCE
	material.iridescence = iridescence;
	material.iridescenceIOR = iridescenceIOR;
	#ifdef USE_IRIDESCENCEMAP
		material.iridescence *= texture2D( iridescenceMap, vIridescenceMapUv ).r;
	#endif
	#ifdef USE_IRIDESCENCE_THICKNESSMAP
		material.iridescenceThickness = (iridescenceThicknessMaximum - iridescenceThicknessMinimum) * texture2D( iridescenceThicknessMap, vIridescenceThicknessMapUv ).g + iridescenceThicknessMinimum;
	#else
		material.iridescenceThickness = iridescenceThicknessMaximum;
	#endif
#endif
#ifdef USE_SHEEN
	material.sheenColor = sheenColor;
	#ifdef USE_SHEEN_COLORMAP
		material.sheenColor *= texture2D( sheenColorMap, vSheenColorMapUv ).rgb;
	#endif
	material.sheenRoughness = clamp( sheenRoughness, 0.0001, 1.0 );
	#ifdef USE_SHEEN_ROUGHNESSMAP
		material.sheenRoughness *= texture2D( sheenRoughnessMap, vSheenRoughnessMapUv ).a;
	#endif
#endif
#ifdef USE_ANISOTROPY
	#ifdef USE_ANISOTROPYMAP
		mat2 anisotropyMat = mat2( anisotropyVector.x, anisotropyVector.y, - anisotropyVector.y, anisotropyVector.x );
		vec3 anisotropyPolar = texture2D( anisotropyMap, vAnisotropyMapUv ).rgb;
		vec2 anisotropyV = anisotropyMat * normalize( 2.0 * anisotropyPolar.rg - vec2( 1.0 ) ) * anisotropyPolar.b;
	#else
		vec2 anisotropyV = anisotropyVector;
	#endif
	material.anisotropy = length( anisotropyV );
	if( material.anisotropy == 0.0 ) {
		anisotropyV = vec2( 1.0, 0.0 );
	} else {
		anisotropyV /= material.anisotropy;
		material.anisotropy = saturate( material.anisotropy );
	}
	material.alphaT = mix( pow2( material.roughness ), 1.0, pow2( material.anisotropy ) );
	material.anisotropyT = tbn[ 0 ] * anisotropyV.x + tbn[ 1 ] * anisotropyV.y;
	material.anisotropyB = tbn[ 1 ] * anisotropyV.x - tbn[ 0 ] * anisotropyV.y;
#endif`,QC=`uniform sampler2D dfgLUT;
struct PhysicalMaterial {
	vec3 diffuseColor;
	vec3 diffuseContribution;
	vec3 specularColor;
	vec3 specularColorBlended;
	float roughness;
	float metalness;
	float specularF90;
	float dispersion;
	vec2 dfg;
	vec3 multiScatteringCompensation;
	#ifdef USE_RETROREFLECTION
		float retroreflectivity;
	#endif
	#ifdef USE_CLEARCOAT
		float clearcoat;
		float clearcoatRoughness;
		vec3 clearcoatF0;
		float clearcoatF90;
	#endif
	#ifdef USE_IRIDESCENCE
		float iridescence;
		float iridescenceIOR;
		float iridescenceThickness;
		vec3 iridescenceFresnel;
		vec3 iridescenceF0Dielectric;
		vec3 iridescenceF0Metallic;
	#endif
	#ifdef USE_SHEEN
		vec3 sheenColor;
		float sheenRoughness;
	#endif
	#ifdef IOR
		float ior;
	#endif
	#ifdef USE_TRANSMISSION
		float transmission;
		float transmissionAlpha;
		float thickness;
		float attenuationDistance;
		vec3 attenuationColor;
	#endif
	#ifdef USE_ANISOTROPY
		float anisotropy;
		float alphaT;
		vec3 anisotropyT;
		vec3 anisotropyB;
	#endif
};
vec3 clearcoatSpecularDirect = vec3( 0.0 );
vec3 clearcoatSpecularIndirect = vec3( 0.0 );
vec3 sheenSpecularDirect = vec3( 0.0 );
vec3 sheenSpecularIndirect = vec3(0.0 );
vec3 Schlick_to_F0( const in vec3 f, const in float f90, const in float dotVH ) {
    float x = clamp( 1.0 - dotVH, 0.0, 1.0 );
    float x2 = x * x;
    float x5 = clamp( x * x2 * x2, 0.0, 0.9999 );
    return ( f - vec3( f90 ) * x5 ) / ( 1.0 - x5 );
}
float V_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {
	float a2 = pow2( alpha );
	float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
	float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );
	return 0.5 / max( gv + gl, EPSILON );
}
float D_GGX( const in float alpha, const in float dotNH ) {
	float a2 = pow2( alpha );
	float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0;
	return RECIPROCAL_PI * a2 / pow2( denom );
}
#ifdef USE_ANISOTROPY
	float V_GGX_SmithCorrelated_Anisotropic( const in float alphaT, const in float alphaB, const in float dotTV, const in float dotBV, const in float dotTL, const in float dotBL, const in float dotNV, const in float dotNL ) {
		float gv = dotNL * length( vec3( alphaT * dotTV, alphaB * dotBV, dotNV ) );
		float gl = dotNV * length( vec3( alphaT * dotTL, alphaB * dotBL, dotNL ) );
		return 0.5 / max( gv + gl, EPSILON );
	}
	float D_GGX_Anisotropic( const in float alphaT, const in float alphaB, const in float dotNH, const in float dotTH, const in float dotBH ) {
		float a2 = alphaT * alphaB;
		highp vec3 v = vec3( alphaB * dotTH, alphaT * dotBH, a2 * dotNH );
		highp float v2 = dot( v, v );
		float w2 = a2 / v2;
		return RECIPROCAL_PI * a2 * pow2 ( w2 );
	}
#endif
#ifdef USE_CLEARCOAT
	vec3 BRDF_GGX_Clearcoat( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material) {
		vec3 f0 = material.clearcoatF0;
		float f90 = material.clearcoatF90;
		float roughness = material.clearcoatRoughness;
		float alpha = pow2( roughness );
		vec3 halfDir = normalize( lightDir + viewDir );
		float dotNL = saturate( dot( normal, lightDir ) );
		float dotNV = saturate( dot( normal, viewDir ) );
		float dotNH = saturate( dot( normal, halfDir ) );
		float dotVH = saturate( dot( viewDir, halfDir ) );
		vec3 F = F_Schlick( f0, f90, dotVH );
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
		return F * ( V * D );
	}
#endif
vec3 BRDF_GGX( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 f0 = material.specularColorBlended;
	float f90 = material.specularF90;
	float roughness = material.roughness;
	float alpha = pow2( roughness );
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( f0, f90, dotVH );
	#ifdef USE_IRIDESCENCE
		F = mix( F, material.iridescenceFresnel, material.iridescence );
	#endif
	#ifdef USE_ANISOTROPY
		float dotTL = dot( material.anisotropyT, lightDir );
		float dotTV = dot( material.anisotropyT, viewDir );
		float dotTH = dot( material.anisotropyT, halfDir );
		float dotBL = dot( material.anisotropyB, lightDir );
		float dotBV = dot( material.anisotropyB, viewDir );
		float dotBH = dot( material.anisotropyB, halfDir );
		float V = V_GGX_SmithCorrelated_Anisotropic( material.alphaT, alpha, dotTV, dotBV, dotTL, dotBL, dotNV, dotNL );
		float D = D_GGX_Anisotropic( material.alphaT, alpha, dotNH, dotTH, dotBH );
	#else
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
	#endif
	return F * ( V * D );
}
vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {
	const float LUT_SIZE = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS = 0.5 / LUT_SIZE;
	float dotNV = saturate( dot( N, V ) );
	vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );
	uv = uv * LUT_SCALE + LUT_BIAS;
	return uv;
}
float LTC_ClippedSphereFormFactor( const in vec3 f ) {
	float l = length( f );
	return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );
}
vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {
	float x = dot( v1, v2 );
	float y = abs( x );
	float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;
	float b = 3.4175940 + ( 4.1616724 + y ) * y;
	float v = a / b;
	float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;
	return cross( v1, v2 ) * theta_sintheta;
}
vec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {
	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];
	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];
	vec3 lightNormal = cross( v1, v2 );
	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );
	vec3 T1, T2;
	T1 = normalize( V - N * dot( V, N ) );
	T2 = - cross( N, T1 );
	mat3 mat = mInv * transpose( mat3( T1, T2, N ) );
	vec3 coords[ 4 ];
	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );
	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );
	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );
	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );
	coords[ 0 ] = normalize( coords[ 0 ] );
	coords[ 1 ] = normalize( coords[ 1 ] );
	coords[ 2 ] = normalize( coords[ 2 ] );
	coords[ 3 ] = normalize( coords[ 3 ] );
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );
	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );
	return vec3( result );
}
#if defined( USE_SHEEN )
float D_Charlie( float roughness, float dotNH ) {
	float alpha = pow2( roughness );
	float invAlpha = 1.0 / alpha;
	float cos2h = dotNH * dotNH;
	float sin2h = max( 1.0 - cos2h, 0.0078125 );
	return ( 2.0 + invAlpha ) * pow( sin2h, invAlpha * 0.5 ) / ( 2.0 * PI );
}
float V_Neubelt( float dotNV, float dotNL ) {
	return saturate( 1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) ) );
}
vec3 BRDF_Sheen( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, vec3 sheenColor, const in float sheenRoughness ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float D = D_Charlie( sheenRoughness, dotNH );
	float V = V_Neubelt( dotNV, dotNL );
	return sheenColor * ( D * V );
}
#endif
float IBLSheenBRDF( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	float r2 = roughness * roughness;
	float rInv = 1.0 / ( roughness + 0.1 );
	float a = -1.9362 + 1.0678 * roughness + 0.4573 * r2 - 0.8469 * rInv;
	float b = -0.6014 + 0.5538 * roughness - 0.4670 * r2 - 0.1255 * rInv;
	float DG = exp( a * dotNV + b );
	return saturate( DG );
}
vec3 EnvironmentBRDF( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 fab = texture2D( dfgLUT, vec2( roughness, dotNV ) ).rg;
	return specularColor * fab.x + specularF90 * fab.y;
}
#ifdef USE_IRIDESCENCE
void computeMultiscatteringIridescence( const in vec2 fab, const in vec3 specularColor, const in float specularF90, const in float iridescence, const in vec3 iridescenceF0, inout vec3 singleScatter, inout vec3 multiScatter ) {
#else
void computeMultiscattering( const in vec2 fab, const in vec3 specularColor, const in float specularF90, inout vec3 singleScatter, inout vec3 multiScatter ) {
#endif
	#ifdef USE_IRIDESCENCE
		vec3 Fr = mix( specularColor, iridescenceF0, iridescence );
	#else
		vec3 Fr = specularColor;
	#endif
	vec3 FssEss = Fr * fab.x + specularF90 * fab.y;
	float Ess = fab.x + fab.y;
	float Ems = 1.0 - Ess;
	vec3 Favg = Fr + ( 1.0 - Fr ) * 0.047619;	vec3 Fms = FssEss * Favg / ( 1.0 - Ems * Favg );
	singleScatter += FssEss;
	multiScatter += Fms * Ems;
}
#if NUM_RECT_AREA_LIGHTS > 0
	void RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
		vec3 normal = geometryNormal;
		vec3 viewDir = geometryViewDir;
		vec3 position = geometryPosition;
		vec3 lightPos = rectAreaLight.position;
		vec3 halfWidth = rectAreaLight.halfWidth;
		vec3 halfHeight = rectAreaLight.halfHeight;
		vec3 lightColor = rectAreaLight.color;
		float roughness = material.roughness;
		vec3 rectCoords[ 4 ];
		rectCoords[ 0 ] = lightPos + halfWidth - halfHeight;		rectCoords[ 1 ] = lightPos - halfWidth - halfHeight;
		rectCoords[ 2 ] = lightPos - halfWidth + halfHeight;
		rectCoords[ 3 ] = lightPos + halfWidth + halfHeight;
		vec2 uv = LTC_Uv( normal, viewDir, roughness );
		vec4 t1 = texture2D( ltc_1, uv );
		vec4 t2 = texture2D( ltc_2, uv );
		mat3 mInv = mat3(
			vec3( t1.x, 0, t1.y ),
			vec3(    0, 1,    0 ),
			vec3( t1.z, 0, t1.w )
		);
		vec3 fresnel = ( material.specularColorBlended * t2.x + ( material.specularF90 - material.specularColorBlended ) * t2.y );
		reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );
		reflectedLight.directDiffuse += lightColor * material.diffuseContribution * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );
		#ifdef USE_CLEARCOAT
			vec3 Ncc = geometryClearcoatNormal;
			vec2 uvClearcoat = LTC_Uv( Ncc, viewDir, material.clearcoatRoughness );
			vec4 t1Clearcoat = texture2D( ltc_1, uvClearcoat );
			vec4 t2Clearcoat = texture2D( ltc_2, uvClearcoat );
			mat3 mInvClearcoat = mat3(
				vec3( t1Clearcoat.x, 0, t1Clearcoat.y ),
				vec3(             0, 1,             0 ),
				vec3( t1Clearcoat.z, 0, t1Clearcoat.w )
			);
			vec3 fresnelClearcoat = material.clearcoatF0 * t2Clearcoat.x + ( material.clearcoatF90 - material.clearcoatF0 ) * t2Clearcoat.y;
			clearcoatSpecularDirect += lightColor * fresnelClearcoat * LTC_Evaluate( Ncc, viewDir, position, mInvClearcoat, rectCoords );
		#endif
	}
#endif
void RE_Direct_Physical( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	#ifdef USE_CLEARCOAT
		float dotNLcc = saturate( dot( geometryClearcoatNormal, directLight.direction ) );
		vec3 ccIrradiance = dotNLcc * directLight.color;
		clearcoatSpecularDirect += ccIrradiance * BRDF_GGX_Clearcoat( directLight.direction, geometryViewDir, geometryClearcoatNormal, material );
	#endif
	#ifdef USE_SHEEN
 
 		sheenSpecularDirect += irradiance * BRDF_Sheen( directLight.direction, geometryViewDir, geometryNormal, material.sheenColor, material.sheenRoughness );
 
 		float sheenAlbedoV = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
 		float sheenAlbedoL = IBLSheenBRDF( geometryNormal, directLight.direction, material.sheenRoughness );
 
 		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * max( sheenAlbedoV, sheenAlbedoL );
 
 		irradiance *= sheenEnergyComp;
 
 	#endif
	vec3 specularBRDF = BRDF_GGX( directLight.direction, geometryViewDir, geometryNormal, material );
	#ifdef USE_RETROREFLECTION
		vec3 retroViewDir = reflect( - geometryViewDir, geometryNormal );
		vec3 retroSpecularBRDF = BRDF_GGX( directLight.direction, retroViewDir, geometryNormal, material );
		specularBRDF = mix( specularBRDF, retroSpecularBRDF, saturate( material.retroreflectivity ) );
	#endif
	reflectedLight.directSpecular += irradiance * specularBRDF * material.multiScatteringCompensation;
	vec3 halfDir = normalize( directLight.direction + geometryViewDir );
	float dotVH = saturate( dot( geometryViewDir, halfDir ) );
	vec3 F = F_Schlick( material.specularColor, material.specularF90, dotVH );
	#ifdef USE_RETROREFLECTION
		vec3 retroHalfDir = normalize( directLight.direction + retroViewDir );
		float dotRetroVH = saturate( dot( retroViewDir, retroHalfDir ) );
		vec3 retroF = F_Schlick( material.specularColor, material.specularF90, dotRetroVH );
		F = mix( F, retroF, saturate( material.retroreflectivity ) );
	#endif
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseContribution ) * ( 1.0 - F );
}
void RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 singleScattering = vec3( 0.0 );
	vec3 multiScattering = vec3( 0.0 );
	#ifdef USE_IRIDESCENCE
		computeMultiscatteringIridescence( material.dfg, material.specularColor, material.specularF90, material.iridescence, material.iridescenceF0Dielectric, singleScattering, multiScattering );
	#else
		computeMultiscattering( material.dfg, material.specularColor, material.specularF90, singleScattering, multiScattering );
	#endif
	vec3 diffuse = irradiance * BRDF_Lambert( material.diffuseContribution ) * ( 1.0 - singleScattering - multiScattering );
	#ifdef USE_SHEEN
		float sheenAlbedo = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
		sheenSpecularIndirect += irradiance * material.sheenColor * sheenAlbedo * RECIPROCAL_PI;
		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * sheenAlbedo;
		diffuse *= sheenEnergyComp;
	#endif
	reflectedLight.indirectDiffuse += diffuse;
}
void RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {
	#ifdef USE_CLEARCOAT
		clearcoatSpecularIndirect += clearcoatRadiance * EnvironmentBRDF( geometryClearcoatNormal, geometryViewDir, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularIndirect += irradiance * material.sheenColor * IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness ) * RECIPROCAL_PI;
 	#endif
	vec3 singleScatteringDielectric = vec3( 0.0 );
	vec3 multiScatteringDielectric = vec3( 0.0 );
	vec3 singleScatteringMetallic = vec3( 0.0 );
	vec3 multiScatteringMetallic = vec3( 0.0 );
	#ifdef USE_IRIDESCENCE
		computeMultiscatteringIridescence( material.dfg, material.specularColor, material.specularF90, material.iridescence, material.iridescenceF0Dielectric, singleScatteringDielectric, multiScatteringDielectric );
		computeMultiscatteringIridescence( material.dfg, material.diffuseColor, material.specularF90, material.iridescence, material.iridescenceF0Metallic, singleScatteringMetallic, multiScatteringMetallic );
	#else
		computeMultiscattering( material.dfg, material.specularColor, material.specularF90, singleScatteringDielectric, multiScatteringDielectric );
		computeMultiscattering( material.dfg, material.diffuseColor, material.specularF90, singleScatteringMetallic, multiScatteringMetallic );
	#endif
	vec3 singleScattering = mix( singleScatteringDielectric, singleScatteringMetallic, material.metalness );
	vec3 multiScattering = mix( multiScatteringDielectric, multiScatteringMetallic, material.metalness );
	vec3 totalScatteringDielectric = singleScatteringDielectric + multiScatteringDielectric;
	vec3 diffuse = material.diffuseContribution * ( 1.0 - totalScatteringDielectric );
	vec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;
	vec3 indirectSpecular = radiance * singleScattering;
	indirectSpecular += multiScattering * cosineWeightedIrradiance;
	vec3 indirectDiffuse = diffuse * cosineWeightedIrradiance;
	#ifdef USE_SHEEN
		float sheenAlbedo = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * sheenAlbedo;
		indirectSpecular *= sheenEnergyComp;
		indirectDiffuse *= sheenEnergyComp;
	#endif
	reflectedLight.indirectSpecular += indirectSpecular;
	reflectedLight.indirectDiffuse += indirectDiffuse;
}
#define RE_Direct				RE_Direct_Physical
#define RE_Direct_RectArea		RE_Direct_RectArea_Physical
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Physical
#define RE_IndirectSpecular		RE_IndirectSpecular_Physical
float computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {
	return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );
}`,ZC=`
vec3 geometryPosition = - vViewPosition;
vec3 geometryNormal = normal;
vec3 geometryViewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );
vec3 geometryClearcoatNormal = vec3( 0.0 );
#ifdef USE_CLEARCOAT
	geometryClearcoatNormal = clearcoatNormal;
#endif
#ifdef USE_IRIDESCENCE
	float dotNVi = saturate( dot( normal, geometryViewDir ) );
	if ( material.iridescenceThickness == 0.0 ) {
		material.iridescence = 0.0;
	} else {
		material.iridescence = saturate( material.iridescence );
	}
	if ( material.iridescence > 0.0 ) {
		vec3 iridescenceFresnelDielectric = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.specularColor );
		vec3 iridescenceFresnelMetallic = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.diffuseColor );
		material.iridescenceFresnel = mix( iridescenceFresnelDielectric, iridescenceFresnelMetallic, material.metalness );
		material.iridescenceF0Dielectric = Schlick_to_F0( iridescenceFresnelDielectric, 1.0, dotNVi );
		material.iridescenceF0Metallic = Schlick_to_F0( iridescenceFresnelMetallic, 1.0, dotNVi );
	}
#endif
#ifdef STANDARD
	float dotNVms = saturate( dot( geometryNormal, geometryViewDir ) );
	material.dfg = texture2D( dfgLUT, vec2( material.roughness, dotNVms ) ).rg;
	#if ( NUM_SUN_LIGHTS > 0 || NUM_DIR_LIGHTS > 0 || NUM_POINT_LIGHTS > 0 || NUM_SPOT_LIGHTS > 0 )
		float EssMs = material.dfg.x + material.dfg.y;
		material.multiScatteringCompensation = 1.0 + material.specularColorBlended * ( 1.0 / EssMs - 1.0 );
	#endif
#endif
IncidentLight directLight;
#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )
	PointLight pointLight;
	#if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {
		pointLight = pointLights[ i ];
		getPointLightInfo( pointLight, geometryPosition, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS ) && ( defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_BASIC ) )
		pointLightShadow = pointLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowIntensity, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )
	SpotLight spotLight;
	vec4 spotColor;
	vec3 spotLightCoord;
	bool inSpotLightMap;
	#if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {
		spotLight = spotLights[ i ];
		getSpotLightInfo( spotLight, geometryPosition, directLight );
		#if ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#define SPOT_LIGHT_MAP_INDEX UNROLLED_LOOP_INDEX
		#elif ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		#define SPOT_LIGHT_MAP_INDEX NUM_SPOT_LIGHT_MAPS
		#else
		#define SPOT_LIGHT_MAP_INDEX ( UNROLLED_LOOP_INDEX - NUM_SPOT_LIGHT_SHADOWS + NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#endif
		#if ( SPOT_LIGHT_MAP_INDEX < NUM_SPOT_LIGHT_MAPS )
			spotLightCoord = vSpotLightCoord[ i ].xyz / vSpotLightCoord[ i ].w;
			inSpotLightMap = all( lessThan( abs( spotLightCoord * 2. - 1. ), vec3( 1.0 ) ) );
			spotColor = texture2D( spotLightMap[ SPOT_LIGHT_MAP_INDEX ], spotLightCoord.xy );
			directLight.color = inSpotLightMap ? directLight.color * spotColor.rgb : directLight.color;
		#endif
		#undef SPOT_LIGHT_MAP_INDEX
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		spotLightShadow = spotLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowIntensity, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_SUN_LIGHTS > 0 ) && defined( RE_Direct )
	SunLight sunLight;
	#if defined( USE_SHADOWMAP ) && NUM_SUN_LIGHT_SHADOWS > 0
	SunLightShadow sunLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SUN_LIGHTS; i ++ ) {
		sunLight = sunLights[ i ];
		getSunLightInfo( sunLight, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SUN_LIGHT_SHADOWS )
		sunLightShadow = sunLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getSunShadow( sunShadowMap[ i ], sunLightShadow, UNROLLED_LOOP_INDEX ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )
	DirectionalLight directionalLight;
	#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {
		directionalLight = directionalLights[ i ];
		getDirectionalLightInfo( directionalLight, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
		directionalLightShadow = directionalLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )
	RectAreaLight rectAreaLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {
		rectAreaLight = rectAreaLights[ i ];
		RE_Direct_RectArea( rectAreaLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if defined( RE_IndirectDiffuse )
	vec3 iblIrradiance = vec3( 0.0 );
	vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );
	#if defined( USE_LIGHT_PROBES )
		irradiance += getLightProbeIrradiance( lightProbe, geometryNormal );
	#endif
	#if ( NUM_HEMI_LIGHTS > 0 )
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {
			irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometryNormal );
		}
		#pragma unroll_loop_end
	#endif
	#ifdef USE_LIGHT_PROBES_GRID
		vec3 probeWorldPos = ( ( vec4( geometryPosition, 1.0 ) - viewMatrix[ 3 ] ) * viewMatrix ).xyz;
		vec3 probeWorldNormal = transformNormalByInverseViewMatrix( geometryNormal, viewMatrix );
		irradiance += getLightProbeGridIrradiance( probeWorldPos, probeWorldNormal );
	#endif
#endif
#if defined( RE_IndirectSpecular )
	vec3 radiance = vec3( 0.0 );
	vec3 clearcoatRadiance = vec3( 0.0 );
#endif`,KC=`#if defined( RE_IndirectDiffuse )
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		vec3 lightMapIrradiance = lightMapTexel.rgb * lightMapIntensity;
		irradiance += lightMapIrradiance;
	#endif
	#if defined( USE_ENVMAP ) && defined( ENVMAP_TYPE_CUBE_UV )
		#if defined( STANDARD ) || defined( LAMBERT ) || defined( PHONG )
			iblIrradiance += getIBLIrradiance( geometryNormal );
		#endif
	#endif
#endif
#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )
	#ifdef USE_ANISOTROPY
		vec3 iblRadiance = getIBLAnisotropyRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );
	#else
		vec3 iblRadiance = getIBLRadiance( geometryViewDir, geometryNormal, material.roughness );
	#endif
	#ifdef USE_RETROREFLECTION
		#ifdef USE_ANISOTROPY
			vec3 retroIBLRadiance = getIBLAnisotropyRetroRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );
		#else
			vec3 retroIBLRadiance = getIBLRetroRadiance( geometryViewDir, geometryNormal, material.roughness );
		#endif
		iblRadiance = mix( iblRadiance, retroIBLRadiance, saturate( material.retroreflectivity ) );
	#endif
	radiance += iblRadiance;
	#ifdef USE_CLEARCOAT
		clearcoatRadiance += getIBLRadiance( geometryViewDir, geometryClearcoatNormal, material.clearcoatRoughness );
	#endif
#endif`,JC=`#if defined( RE_IndirectDiffuse )
	#if defined( LAMBERT ) || defined( PHONG )
		irradiance += iblIrradiance;
	#endif
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`,jC=`#ifdef USE_LIGHT_PROBES_GRID
uniform highp sampler3D probesSH;
uniform vec3 probesMin;
uniform vec3 probesMax;
uniform vec3 probesResolution;
vec3 getLightProbeGridIrradiance( vec3 worldPos, vec3 worldNormal ) {
	vec3 res = probesResolution;
	vec3 gridRange = probesMax - probesMin;
	vec3 resMinusOne = res - 1.0;
	vec3 probeSpacing = gridRange / resMinusOne;
	vec3 samplePos = worldPos + worldNormal * probeSpacing * 0.5;
	vec3 uvw = clamp( ( samplePos - probesMin ) / gridRange, 0.0, 1.0 );
	uvw = uvw * resMinusOne / res + 0.5 / res;
	float nz          = res.z;
	float paddedSlices = nz + 2.0;
	float atlasDepth  = 7.0 * paddedSlices;
	float uvZBase     = uvw.z * nz + 1.0;
	vec4 s0 = texture( probesSH, vec3( uvw.xy, ( uvZBase                       ) / atlasDepth ) );
	vec4 s1 = texture( probesSH, vec3( uvw.xy, ( uvZBase +       paddedSlices   ) / atlasDepth ) );
	vec4 s2 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 2.0 * paddedSlices   ) / atlasDepth ) );
	vec4 s3 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 3.0 * paddedSlices   ) / atlasDepth ) );
	vec4 s4 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 4.0 * paddedSlices   ) / atlasDepth ) );
	vec4 s5 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 5.0 * paddedSlices   ) / atlasDepth ) );
	vec4 s6 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 6.0 * paddedSlices   ) / atlasDepth ) );
	vec3 c0 = s0.xyz;
	vec3 c1 = vec3( s0.w, s1.xy );
	vec3 c2 = vec3( s1.zw, s2.x );
	vec3 c3 = s2.yzw;
	vec3 c4 = s3.xyz;
	vec3 c5 = vec3( s3.w, s4.xy );
	vec3 c6 = vec3( s4.zw, s5.x );
	vec3 c7 = s5.yzw;
	vec3 c8 = s6.xyz;
	float x = worldNormal.x, y = worldNormal.y, z = worldNormal.z;
	vec3 result = c0 * 0.886227;
	result += c1 * 2.0 * 0.511664 * y;
	result += c2 * 2.0 * 0.511664 * z;
	result += c3 * 2.0 * 0.511664 * x;
	result += c4 * 2.0 * 0.429043 * x * y;
	result += c5 * 2.0 * 0.429043 * y * z;
	result += c6 * ( 0.743125 * z * z - 0.247708 );
	result += c7 * 2.0 * 0.429043 * x * z;
	result += c8 * 0.429043 * ( x * x - y * y );
	return max( result, vec3( 0.0 ) );
}
#endif`,$C=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`,e2=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,t2=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,n2=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	vFragDepth = 1.0 + gl_Position.w;
	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
#endif`,i2=`#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`,s2=`#ifdef USE_MAP
	uniform sampler2D map;
#endif`,r2=`#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
	#if defined( USE_POINTS_UV )
		vec2 uv = vUv;
	#else
		vec2 uv = ( uvTransform * vec3( gl_PointCoord.x, 1.0 - gl_PointCoord.y, 1 ) ).xy;
	#endif
#endif
#ifdef USE_MAP
	diffuseColor *= texture2D( map, uv );
#endif
#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, uv ).g;
#endif`,a2=`#if defined( USE_POINTS_UV )
	varying vec2 vUv;
#else
	#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
		uniform mat3 uvTransform;
	#endif
#endif
#ifdef USE_MAP
	uniform sampler2D map;
#endif
#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,o2=`float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`,l2=`#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`,c2=`#ifdef USE_INSTANCING_MORPH
	float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;
	}
#endif`,u2=`#if defined( USE_MORPHCOLORS )
	vColor *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`,f2=`#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,h2=`#ifdef USE_MORPHTARGETS
	#ifndef USE_INSTANCING_MORPH
		uniform float morphTargetBaseInfluence;
		uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	#endif
	uniform sampler2DArray morphTargetsTexture;
	uniform ivec2 morphTargetsTextureSize;
	vec4 getMorph( const in int vertexIndex, const in int morphTargetIndex, const in int offset ) {
		int texelIndex = vertexIndex * MORPHTARGETS_TEXTURE_STRIDE + offset;
		int y = texelIndex / morphTargetsTextureSize.x;
		int x = texelIndex - y * morphTargetsTextureSize.x;
		ivec3 morphUV = ivec3( x, y, morphTargetIndex );
		return texelFetch( morphTargetsTexture, morphUV, 0 );
	}
#endif`,d2=`#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,p2=`float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
#ifdef FLAT_SHADED
	vec3 fdx = dFdx( vViewPosition );
	vec3 fdy = dFdy( vViewPosition );
	vec3 normal = normalize( cross( fdx, fdy ) );
#else
	vec3 normal = normalize( vNormal );
	#ifdef DOUBLE_SIDED
		normal *= faceDirection;
	#endif
#endif
#if defined( USE_NORMALMAP_TANGENTSPACE ) || defined( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY )
	#ifdef USE_TANGENT
		mat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn = getTangentFrame( - vViewPosition, normal,
		#if defined( USE_NORMALMAP )
			vNormalMapUv
		#elif defined( USE_CLEARCOAT_NORMALMAP )
			vClearcoatNormalMapUv
		#else
			vUv
		#endif
		);
	#endif
	#ifdef DOUBLE_SIDED
		tbn[0] *= faceDirection;
		tbn[1] *= faceDirection;
	#endif
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	#ifdef USE_TANGENT
		mat3 tbn2 = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn2 = getTangentFrame( - vViewPosition, normal, vClearcoatNormalMapUv );
	#endif
	#ifdef DOUBLE_SIDED
		tbn2[0] *= faceDirection;
		tbn2[1] *= faceDirection;
	#endif
#endif
vec3 nonPerturbedNormal = normal;`,m2=`#ifdef USE_NORMALMAP_OBJECTSPACE
	normal = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#ifdef FLIP_SIDED
		normal = - normal;
	#endif
	#ifdef DOUBLE_SIDED
		normal = normal * faceDirection;
	#endif
	normal = normalize( normalMatrix * normal );
#elif defined( USE_NORMALMAP_TANGENTSPACE )
	vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#if defined( USE_PACKED_NORMALMAP )
		mapN = vec3( mapN.xy, sqrt( saturate( 1.0 - dot( mapN.xy, mapN.xy ) ) ) );
	#endif
	mapN.xy *= normalScale;
	normal = normalize( tbn * mapN );
#elif defined( USE_BUMPMAP )
	normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );
#endif`,g2=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,v2=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,x2=`#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
		#ifdef FLIP_SIDED
			vBitangent = - vBitangent;
		#endif
	#endif
#endif`,y2=`#ifdef USE_NORMALMAP
	uniform sampler2D normalMap;
	uniform vec2 normalScale;
#endif
#ifdef USE_NORMALMAP_OBJECTSPACE
	uniform mat3 normalMatrix;
#endif
#if ! defined ( USE_TANGENT ) && ( defined ( USE_NORMALMAP_TANGENTSPACE ) || defined ( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY ) )
	mat3 getTangentFrame( vec3 eye_pos, vec3 surf_norm, vec2 uv ) {
		vec3 q0 = dFdx( eye_pos.xyz );
		vec3 q1 = dFdy( eye_pos.xyz );
		vec2 st0 = dFdx( uv.st );
		vec2 st1 = dFdy( uv.st );
		vec3 N = surf_norm;
		vec3 q1perp = cross( q1, N );
		vec3 q0perp = cross( N, q0 );
		vec3 T = q1perp * st0.x + q0perp * st1.x;
		vec3 B = q1perp * st0.y + q0perp * st1.y;
		float det = max( dot( T, T ), dot( B, B ) );
		float scale = ( det == 0.0 ) ? 0.0 : inversesqrt( det );
		return mat3( T * scale, B * scale, N );
	}
#endif`,_2=`#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`,S2=`#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`,A2=`#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`,M2=`#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`,E2=`#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`,T2=`vec3 packNormalToRGB( const in vec3 normal ) {
	return normalize( normal ) * 0.5 + 0.5;
}
vec3 unpackRGBToNormal( const in vec3 rgb ) {
	return 2.0 * rgb.xyz - 1.0;
}
const float PackUpscale = 256. / 255.;const float UnpackDownscale = 255. / 256.;const float ShiftRight8 = 1. / 256.;
const float Inv255 = 1. / 255.;
const vec4 PackFactors = vec4( 1.0, 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0 );
const vec2 UnpackFactors2 = vec2( UnpackDownscale, 1.0 / PackFactors.g );
const vec3 UnpackFactors3 = vec3( UnpackDownscale / PackFactors.rg, 1.0 / PackFactors.b );
const vec4 UnpackFactors4 = vec4( UnpackDownscale / PackFactors.rgb, 1.0 / PackFactors.a );
vec4 packDepthToRGBA( const in float v ) {
	if( v <= 0.0 )
		return vec4( 0., 0., 0., 0. );
	if( v >= 1.0 )
		return vec4( 1., 1., 1., 1. );
	float vuf;
	float af = modf( v * PackFactors.a, vuf );
	float bf = modf( vuf * ShiftRight8, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec4( vuf * Inv255, gf * PackUpscale, bf * PackUpscale, af );
}
vec3 packDepthToRGB( const in float v ) {
	if( v <= 0.0 )
		return vec3( 0., 0., 0. );
	if( v >= 1.0 )
		return vec3( 1., 1., 1. );
	float vuf;
	float bf = modf( v * PackFactors.b, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec3( vuf * Inv255, gf * PackUpscale, bf );
}
vec2 packDepthToRG( const in float v ) {
	if( v <= 0.0 )
		return vec2( 0., 0. );
	if( v >= 1.0 )
		return vec2( 1., 1. );
	float vuf;
	float gf = modf( v * 256., vuf );
	return vec2( vuf * Inv255, gf );
}
float unpackRGBAToDepth( const in vec4 v ) {
	return dot( v, UnpackFactors4 );
}
float unpackRGBToDepth( const in vec3 v ) {
	return dot( v, UnpackFactors3 );
}
float unpackRGToDepth( const in vec2 v ) {
	return v.r * UnpackFactors2.r + v.g * UnpackFactors2.g;
}
vec4 pack2HalfToRGBA( const in vec2 v ) {
	vec4 r = vec4( v.x, fract( v.x * 255.0 ), v.y, fract( v.y * 255.0 ) );
	return vec4( r.x - r.y / 255.0, r.y, r.z - r.w / 255.0, r.w );
}
vec2 unpackRGBATo2Half( const in vec4 v ) {
	return vec2( v.x + ( v.y / 255.0 ), v.z + ( v.w / 255.0 ) );
}
float viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {
	return ( viewZ + near ) / ( near - far );
}
float orthographicDepthToViewZ( const in float depth, const in float near, const in float far ) {
	#ifdef USE_REVERSED_DEPTH_BUFFER
	
		return depth * ( far - near ) - far;
	#else
		return depth * ( near - far ) - near;
	#endif
}
float viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {
	return ( ( near + viewZ ) * far ) / ( ( far - near ) * viewZ );
}
float perspectiveDepthToViewZ( const in float depth, const in float near, const in float far ) {
	
	#ifdef USE_REVERSED_DEPTH_BUFFER
		return ( near * far ) / ( ( near - far ) * depth - near );
	#else
		return ( near * far ) / ( ( far - near ) * depth - far );
	#endif
}`,b2=`#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`,w2=`vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`,C2=`#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`,R2=`#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`,D2=`float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`,U2=`#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`,I2=`#if NUM_SPOT_LIGHT_COORDS > 0
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if NUM_SPOT_LIGHT_MAPS > 0
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_SUN_LIGHT_SHADOWS > 0
		#define SUN_LIGHT_CASCADES 2
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform sampler2DShadow sunShadowMap[ NUM_SUN_LIGHT_SHADOWS ];
		#else
			uniform sampler2D sunShadowMap[ NUM_SUN_LIGHT_SHADOWS ];
		#endif
		uniform mat4 sunShadowMatrix[ NUM_SUN_LIGHT_SHADOWS * SUN_LIGHT_CASCADES ];
		uniform vec4 sunShadowCascade[ NUM_SUN_LIGHT_SHADOWS * SUN_LIGHT_CASCADES ];
		varying vec4 vSunShadowWorldPosition;
		varying vec3 vSunShadowWorldNormal;
		struct SunLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SunLightShadow sunLightShadows[ NUM_SUN_LIGHT_SHADOWS ];
	#endif
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform sampler2DShadow directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		#else
			uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		#endif
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform sampler2DShadow spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		#else
			uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		#endif
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform samplerCubeShadow pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		#elif defined( SHADOWMAP_TYPE_BASIC )
			uniform samplerCube pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		#endif
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
	#if defined( SHADOWMAP_TYPE_PCF )
		float interleavedGradientNoise( vec2 position ) {
			return fract( 52.9829189 * fract( dot( position, vec2( 0.06711056, 0.00583715 ) ) ) );
		}
		vec2 vogelDiskSample( int sampleIndex, int samplesCount, float phi ) {
			const float goldenAngle = 2.399963229728653;
			float r = sqrt( ( float( sampleIndex ) + 0.5 ) / float( samplesCount ) );
			float theta = float( sampleIndex ) * goldenAngle + phi;
			return vec2( cos( theta ), sin( theta ) ) * r;
		}
	#endif
	#if defined( SHADOWMAP_TYPE_PCF )
		float getShadow( sampler2DShadow shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
			float shadow = 1.0;
			shadowCoord.xyz /= shadowCoord.w;
			shadowCoord.z += shadowBias;
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
				float radius = shadowRadius * texelSize.x;
				float phi = interleavedGradientNoise( gl_FragCoord.xy ) * PI2;
				shadow = (
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 0, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 1, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 2, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 3, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 4, 5, phi ) * radius, shadowCoord.z ) )
				) * 0.2;
			}
			return mix( 1.0, shadow, shadowIntensity );
		}
	#elif defined( SHADOWMAP_TYPE_VSM )
		float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
			float shadow = 1.0;
			shadowCoord.xyz /= shadowCoord.w;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				shadowCoord.z -= shadowBias;
			#else
				shadowCoord.z += shadowBias;
			#endif
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				vec2 distribution = texture2D( shadowMap, shadowCoord.xy ).rg;
				float mean = distribution.x;
				float variance = distribution.y * distribution.y;
				#ifdef USE_REVERSED_DEPTH_BUFFER
					float hard_shadow = step( mean, shadowCoord.z );
				#else
					float hard_shadow = step( shadowCoord.z, mean );
				#endif
				
				if ( hard_shadow == 1.0 ) {
					shadow = 1.0;
				} else {
					variance = max( variance, 0.0000001 );
					float d = shadowCoord.z - mean;
					float p_max = variance / ( variance + d * d );
					p_max = clamp( ( p_max - 0.3 ) / 0.65, 0.0, 1.0 );
					shadow = max( hard_shadow, p_max );
				}
			}
			return mix( 1.0, shadow, shadowIntensity );
		}
	#else
		float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
			float shadow = 1.0;
			shadowCoord.xyz /= shadowCoord.w;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				shadowCoord.z -= shadowBias;
			#else
				shadowCoord.z += shadowBias;
			#endif
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				float depth = texture2D( shadowMap, shadowCoord.xy ).r;
				#ifdef USE_REVERSED_DEPTH_BUFFER
					shadow = step( depth, shadowCoord.z );
				#else
					shadow = step( shadowCoord.z, depth );
				#endif
			}
			return mix( 1.0, shadow, shadowIntensity );
		}
	#endif
	#if NUM_SUN_LIGHT_SHADOWS > 0
		float getSunShadow(
			#if defined( SHADOWMAP_TYPE_PCF )
				sampler2DShadow shadowMap,
			#else
				sampler2D shadowMap,
			#endif
			SunLightShadow sunLightShadow,
			int shadowIndex
		) {
			vec4 shadowWorldPosition = vec4( vSunShadowWorldPosition.xyz + vSunShadowWorldNormal * sunLightShadow.shadowNormalBias, 1.0 );
			float viewDepth = vSunShadowWorldPosition.w;
			int cascadeOffset = shadowIndex * SUN_LIGHT_CASCADES;
			float shadow = 1.0;
			for ( int i = SUN_LIGHT_CASCADES - 1; i >= 0; i -- ) {
				vec4 cascade = sunShadowCascade[ cascadeOffset + i ];
				if ( viewDepth >= cascade.x && viewDepth < cascade.y ) {
					float cascadeShadow = getShadow(
						shadowMap,
						sunLightShadow.shadowMapSize,
						sunLightShadow.shadowIntensity,
						sunLightShadow.shadowBias,
						sunLightShadow.shadowRadius,
						sunShadowMatrix[ cascadeOffset + i ] * shadowWorldPosition
					);
					shadow = mix( cascadeShadow, shadow, smoothstep( cascade.z, cascade.y, viewDepth ) );
				}
			}
			return shadow;
		}
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
	#if defined( SHADOWMAP_TYPE_PCF )
	float getPointShadow( samplerCubeShadow shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;
		vec3 bd3D = normalize( lightToPosition );
		vec3 absVec = abs( lightToPosition );
		float viewSpaceZ = max( max( absVec.x, absVec.y ), absVec.z );
		if ( viewSpaceZ - shadowCameraFar <= 0.0 && viewSpaceZ - shadowCameraNear >= 0.0 ) {
			#ifdef USE_REVERSED_DEPTH_BUFFER
				float dp = ( shadowCameraNear * ( shadowCameraFar - viewSpaceZ ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );
				dp -= shadowBias;
			#else
				float dp = ( shadowCameraFar * ( viewSpaceZ - shadowCameraNear ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );
				dp += shadowBias;
			#endif
			float texelSize = shadowRadius / shadowMapSize.x;
			vec3 absDir = abs( bd3D );
			vec3 tangent = absDir.x > absDir.z ? vec3( 0.0, 1.0, 0.0 ) : vec3( 1.0, 0.0, 0.0 );
			tangent = normalize( cross( bd3D, tangent ) );
			vec3 bitangent = cross( bd3D, tangent );
			float phi = interleavedGradientNoise( gl_FragCoord.xy ) * PI2;
			vec2 sample0 = vogelDiskSample( 0, 5, phi );
			vec2 sample1 = vogelDiskSample( 1, 5, phi );
			vec2 sample2 = vogelDiskSample( 2, 5, phi );
			vec2 sample3 = vogelDiskSample( 3, 5, phi );
			vec2 sample4 = vogelDiskSample( 4, 5, phi );
			shadow = (
				texture( shadowMap, vec4( bd3D + ( tangent * sample0.x + bitangent * sample0.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample1.x + bitangent * sample1.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample2.x + bitangent * sample2.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample3.x + bitangent * sample3.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample4.x + bitangent * sample4.y ) * texelSize, dp ) )
			) * 0.2;
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	#elif defined( SHADOWMAP_TYPE_BASIC )
	float getPointShadow( samplerCube shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;
		vec3 absVec = abs( lightToPosition );
		float viewSpaceZ = max( max( absVec.x, absVec.y ), absVec.z );
		if ( viewSpaceZ - shadowCameraFar <= 0.0 && viewSpaceZ - shadowCameraNear >= 0.0 ) {
			float dp = ( shadowCameraFar * ( viewSpaceZ - shadowCameraNear ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );
			dp += shadowBias;
			vec3 bd3D = normalize( lightToPosition );
			float depth = textureCube( shadowMap, bd3D ).r;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				depth = 1.0 - depth;
			#endif
			shadow = step( dp, depth );
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	#endif
	#endif
#endif`,B2=`#if NUM_SPOT_LIGHT_COORDS > 0
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_SUN_LIGHT_SHADOWS > 0
		varying vec4 vSunShadowWorldPosition;
		varying vec3 vSunShadowWorldNormal;
	#endif
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
#endif`,N2=`#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_SUN_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
	#ifdef HAS_NORMAL
		vec3 shadowWorldNormal = transformNormalByInverseViewMatrix( transformedNormal, viewMatrix );
	#else
		vec3 shadowWorldNormal = vec3( 0.0 );
	#endif
	vec4 shadowWorldPosition;
#endif
#if defined( USE_SHADOWMAP )
	#if NUM_SUN_LIGHT_SHADOWS > 0
		vSunShadowWorldPosition = vec4( worldPosition.xyz, - mvPosition.z );
		vSunShadowWorldNormal = shadowWorldNormal;
	#endif
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * directionalLightShadows[ i ].shadowNormalBias, 0 );
			vDirectionalShadowCoord[ i ] = directionalShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * pointLightShadows[ i ].shadowNormalBias, 0 );
			vPointShadowCoord[ i ] = pointShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
#endif
#if NUM_SPOT_LIGHT_COORDS > 0
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_COORDS; i ++ ) {
		shadowWorldPosition = worldPosition;
		#if ( defined( USE_SHADOWMAP ) && UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
			shadowWorldPosition.xyz += shadowWorldNormal * spotLightShadows[ i ].shadowNormalBias;
		#endif
		vSpotLightCoord[ i ] = spotLightMatrix[ i ] * shadowWorldPosition;
	}
	#pragma unroll_loop_end
#endif`,P2=`float getShadowMask() {
	float shadow = 1.0;
	#ifdef USE_SHADOWMAP
	#if NUM_SUN_LIGHT_SHADOWS > 0
	SunLightShadow sunLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SUN_LIGHT_SHADOWS; i ++ ) {
		sunLight = sunLightShadows[ i ];
		shadow *= receiveShadow ? getSunShadow( sunShadowMap[ i ], sunLight, UNROLLED_LOOP_INDEX ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
		directionalLight = directionalLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( directionalShadowMap[ i ], directionalLight.shadowMapSize, directionalLight.shadowIntensity, directionalLight.shadowBias, directionalLight.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; i ++ ) {
		spotLight = spotLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( spotShadowMap[ i ], spotLight.shadowMapSize, spotLight.shadowIntensity, spotLight.shadowBias, spotLight.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0 && ( defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_BASIC ) )
	PointLightShadow pointLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
		pointLight = pointLightShadows[ i ];
		shadow *= receiveShadow ? getPointShadow( pointShadowMap[ i ], pointLight.shadowMapSize, pointLight.shadowIntensity, pointLight.shadowBias, pointLight.shadowRadius, vPointShadowCoord[ i ], pointLight.shadowCameraNear, pointLight.shadowCameraFar ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#endif
	return shadow;
}`,L2=`#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`,O2=`#ifdef USE_SKINNING
	uniform mat4 bindMatrix;
	uniform mat4 bindMatrixInverse;
	uniform highp sampler2D boneTexture;
	mat4 getBoneMatrix( const in float i ) {
		int size = textureSize( boneTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( boneTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( boneTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( boneTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( boneTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
#endif`,F2=`#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`,z2=`#ifdef USE_SKINNING
	mat4 skinMatrix = mat4( 0.0 );
	skinMatrix += skinWeight.x * boneMatX;
	skinMatrix += skinWeight.y * boneMatY;
	skinMatrix += skinWeight.z * boneMatZ;
	skinMatrix += skinWeight.w * boneMatW;
	skinMatrix = bindMatrixInverse * skinMatrix * bindMatrix;
	objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;
	#ifdef USE_TANGENT
		objectTangent = vec4( skinMatrix * vec4( objectTangent, 0.0 ) ).xyz;
	#endif
#endif`,H2=`float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`,G2=`#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`,V2=`#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`,k2=`#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
uniform float toneMappingExposure;
vec3 LinearToneMapping( vec3 color ) {
	return saturate( toneMappingExposure * color );
}
vec3 ReinhardToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	return saturate( color / ( vec3( 1.0 ) + color ) );
}
vec3 CineonToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	color = max( vec3( 0.0 ), color - 0.004 );
	return pow( ( color * ( 6.2 * color + 0.5 ) ) / ( color * ( 6.2 * color + 1.7 ) + 0.06 ), vec3( 2.2 ) );
}
vec3 RRTAndODTFit( vec3 v ) {
	vec3 a = v * ( v + 0.0245786 ) - 0.000090537;
	vec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;
	return a / b;
}
vec3 ACESFilmicToneMapping( vec3 color ) {
	const mat3 ACESInputMat = mat3(
		vec3( 0.59719, 0.07600, 0.02840 ),		vec3( 0.35458, 0.90834, 0.13383 ),
		vec3( 0.04823, 0.01566, 0.83777 )
	);
	const mat3 ACESOutputMat = mat3(
		vec3(  1.60475, -0.10208, -0.00327 ),		vec3( -0.53108,  1.10813, -0.07276 ),
		vec3( -0.07367, -0.00605,  1.07602 )
	);
	color *= toneMappingExposure / 0.6;
	color = ACESInputMat * color;
	color = RRTAndODTFit( color );
	color = ACESOutputMat * color;
	return saturate( color );
}
const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
	vec3( 1.6605, - 0.1246, - 0.0182 ),
	vec3( - 0.5876, 1.1329, - 0.1006 ),
	vec3( - 0.0728, - 0.0083, 1.1187 )
);
const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
	vec3( 0.6274, 0.0691, 0.0164 ),
	vec3( 0.3293, 0.9195, 0.0880 ),
	vec3( 0.0433, 0.0113, 0.8956 )
);
vec3 agxDefaultContrastApprox( vec3 x ) {
	vec3 x2 = x * x;
	vec3 x4 = x2 * x2;
	return + 15.5 * x4 * x2
		- 40.14 * x4 * x
		+ 31.96 * x4
		- 6.868 * x2 * x
		+ 0.4298 * x2
		+ 0.1191 * x
		- 0.00232;
}
vec3 AgXToneMapping( vec3 color ) {
	const mat3 AgXInsetMatrix = mat3(
		vec3( 0.856627153315983, 0.137318972929847, 0.11189821299995 ),
		vec3( 0.0951212405381588, 0.761241990602591, 0.0767994186031903 ),
		vec3( 0.0482516061458583, 0.101439036467562, 0.811302368396859 )
	);
	const mat3 AgXOutsetMatrix = mat3(
		vec3( 1.1271005818144368, - 0.1413297634984383, - 0.14132976349843826 ),
		vec3( - 0.11060664309660323, 1.157823702216272, - 0.11060664309660294 ),
		vec3( - 0.016493938717834573, - 0.016493938717834257, 1.2519364065950405 )
	);
	const float AgxMinEv = - 12.47393;	const float AgxMaxEv = 4.026069;
	color *= toneMappingExposure;
	color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
	color = AgXInsetMatrix * color;
	color = max( color, 1e-10 );	color = log2( color );
	color = ( color - AgxMinEv ) / ( AgxMaxEv - AgxMinEv );
	color = clamp( color, 0.0, 1.0 );
	color = agxDefaultContrastApprox( color );
	color = AgXOutsetMatrix * color;
	color = pow( max( vec3( 0.0 ), color ), vec3( 2.2 ) );
	color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
	color = clamp( color, 0.0, 1.0 );
	return color;
}
vec3 NeutralToneMapping( vec3 color ) {
	const float StartCompression = 0.8 - 0.04;
	const float Desaturation = 0.15;
	color *= toneMappingExposure;
	float x = min( color.r, min( color.g, color.b ) );
	float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
	color -= offset;
	float peak = max( color.r, max( color.g, color.b ) );
	if ( peak < StartCompression ) return color;
	float d = 1. - StartCompression;
	float newPeak = 1. - d * d / ( peak + d - StartCompression );
	color *= newPeak / peak;
	float g = 1. - 1. / ( Desaturation * ( peak - newPeak ) + 1. );
	return mix( color, vec3( newPeak ), g );
}
vec3 CustomToneMapping( vec3 color ) { return color; }`,W2=`#ifdef USE_TRANSMISSION
	material.transmission = transmission;
	material.transmissionAlpha = 1.0;
	material.thickness = thickness;
	material.attenuationDistance = attenuationDistance;
	material.attenuationColor = attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		material.transmission *= texture2D( transmissionMap, vTransmissionMapUv ).r;
	#endif
	#ifdef USE_THICKNESSMAP
		material.thickness *= texture2D( thicknessMap, vThicknessMapUv ).g;
	#endif
	vec3 pos = vWorldPosition;
	vec3 v = normalize( cameraPosition - pos );
	vec3 n = transformNormalByInverseViewMatrix( normal, viewMatrix );
	vec4 transmitted = getIBLVolumeRefraction(
		n, v, material.roughness, material.diffuseContribution, material.specularColorBlended, material.specularF90,
		pos, modelMatrix, viewMatrix, projectionMatrix, material.dispersion, material.ior, material.thickness,
		material.attenuationColor, material.attenuationDistance );
	material.transmissionAlpha = mix( material.transmissionAlpha, transmitted.a, material.transmission );
	totalDiffuse = mix( totalDiffuse, transmitted.rgb, material.transmission );
#endif`,X2=`#ifdef USE_TRANSMISSION
	uniform float transmission;
	uniform float thickness;
	uniform float attenuationDistance;
	uniform vec3 attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		uniform sampler2D transmissionMap;
	#endif
	#ifdef USE_THICKNESSMAP
		uniform sampler2D thicknessMap;
	#endif
	uniform vec2 transmissionSamplerSize;
	uniform sampler2D transmissionSamplerMap;
	uniform mat4 modelMatrix;
	uniform mat4 projectionMatrix;
	varying vec3 vWorldPosition;
	float w0( float a ) {
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - a + 3.0 ) - 3.0 ) + 1.0 );
	}
	float w1( float a ) {
		return ( 1.0 / 6.0 ) * ( a *  a * ( 3.0 * a - 6.0 ) + 4.0 );
	}
	float w2( float a ){
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - 3.0 * a + 3.0 ) + 3.0 ) + 1.0 );
	}
	float w3( float a ) {
		return ( 1.0 / 6.0 ) * ( a * a * a );
	}
	float g0( float a ) {
		return w0( a ) + w1( a );
	}
	float g1( float a ) {
		return w2( a ) + w3( a );
	}
	float h0( float a ) {
		return - 1.0 + w1( a ) / ( w0( a ) + w1( a ) );
	}
	float h1( float a ) {
		return 1.0 + w3( a ) / ( w2( a ) + w3( a ) );
	}
	vec4 bicubic( sampler2D tex, vec2 uv, vec4 texelSize, float lod ) {
		uv = uv * texelSize.zw + 0.5;
		vec2 iuv = floor( uv );
		vec2 fuv = fract( uv );
		float g0x = g0( fuv.x );
		float g1x = g1( fuv.x );
		float h0x = h0( fuv.x );
		float h1x = h1( fuv.x );
		float h0y = h0( fuv.y );
		float h1y = h1( fuv.y );
		vec2 p0 = ( vec2( iuv.x + h0x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p1 = ( vec2( iuv.x + h1x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p2 = ( vec2( iuv.x + h0x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		vec2 p3 = ( vec2( iuv.x + h1x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		return g0( fuv.y ) * ( g0x * textureLod( tex, p0, lod ) + g1x * textureLod( tex, p1, lod ) ) +
			g1( fuv.y ) * ( g0x * textureLod( tex, p2, lod ) + g1x * textureLod( tex, p3, lod ) );
	}
	vec4 textureBicubic( sampler2D sampler, vec2 uv, float lod ) {
		vec2 fLodSize = vec2( textureSize( sampler, int( lod ) ) );
		vec2 cLodSize = vec2( textureSize( sampler, int( lod + 1.0 ) ) );
		vec2 fLodSizeInv = 1.0 / fLodSize;
		vec2 cLodSizeInv = 1.0 / cLodSize;
		vec4 fSample = bicubic( sampler, uv, vec4( fLodSizeInv, fLodSize ), floor( lod ) );
		vec4 cSample = bicubic( sampler, uv, vec4( cLodSizeInv, cLodSize ), ceil( lod ) );
		return mix( fSample, cSample, fract( lod ) );
	}
	vec3 getVolumeTransmissionRay( const in vec3 n, const in vec3 v, const in float thickness, const in float ior, const in mat4 modelMatrix ) {
		vec3 refractionVector = refract( - v, normalize( n ), 1.0 / ior );
		vec3 modelScale;
		modelScale.x = length( vec3( modelMatrix[ 0 ].xyz ) );
		modelScale.y = length( vec3( modelMatrix[ 1 ].xyz ) );
		modelScale.z = length( vec3( modelMatrix[ 2 ].xyz ) );
		return normalize( refractionVector ) * thickness * modelScale;
	}
	float applyIorToRoughness( const in float roughness, const in float ior ) {
		return roughness * clamp( ior * 2.0 - 2.0, 0.0, 1.0 );
	}
	vec4 getTransmissionSample( const in vec2 fragCoord, const in float roughness, const in float ior ) {
		float lod = log2( transmissionSamplerSize.x ) * applyIorToRoughness( roughness, ior );
		return textureBicubic( transmissionSamplerMap, fragCoord.xy, lod );
	}
	vec3 volumeAttenuation( const in float transmissionDistance, const in vec3 attenuationColor, const in float attenuationDistance ) {
		if ( isinf( attenuationDistance ) ) {
			return vec3( 1.0 );
		} else {
			vec3 attenuationCoefficient = -log( attenuationColor ) / attenuationDistance;
			vec3 transmittance = exp( - attenuationCoefficient * transmissionDistance );			return transmittance;
		}
	}
	vec4 getIBLVolumeRefraction( const in vec3 n, const in vec3 v, const in float roughness, const in vec3 diffuseColor,
		const in vec3 specularColor, const in float specularF90, const in vec3 position, const in mat4 modelMatrix,
		const in mat4 viewMatrix, const in mat4 projMatrix, const in float dispersion, const in float ior, const in float thickness,
		const in vec3 attenuationColor, const in float attenuationDistance ) {
		vec4 transmittedLight;
		vec3 transmittance;
		#ifdef USE_DISPERSION
			float halfSpread = ( ior - 1.0 ) * 0.025 * dispersion;
			vec3 iors = vec3( ior - halfSpread, ior, ior + halfSpread );
			for ( int i = 0; i < 3; i ++ ) {
				vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, iors[ i ], modelMatrix );
				vec3 refractedRayExit = position + transmissionRay;
				vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
				vec2 refractionCoords = ndcPos.xy / ndcPos.w;
				refractionCoords += 1.0;
				refractionCoords /= 2.0;
				vec4 transmissionSample = getTransmissionSample( refractionCoords, roughness, iors[ i ] );
				transmittedLight[ i ] = transmissionSample[ i ];
				transmittedLight.a += transmissionSample.a;
				transmittance[ i ] = diffuseColor[ i ] * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance )[ i ];
			}
			transmittedLight.a /= 3.0;
		#else
			vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, ior, modelMatrix );
			vec3 refractedRayExit = position + transmissionRay;
			vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
			vec2 refractionCoords = ndcPos.xy / ndcPos.w;
			refractionCoords += 1.0;
			refractionCoords /= 2.0;
			transmittedLight = getTransmissionSample( refractionCoords, roughness, ior );
			transmittance = diffuseColor * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance );
		#endif
		vec3 attenuatedColor = transmittance * transmittedLight.rgb;
		vec3 F = EnvironmentBRDF( n, v, specularColor, specularF90, roughness );
		float transmittanceFactor = ( transmittance.r + transmittance.g + transmittance.b ) / 3.0;
		return vec4( ( 1.0 - F ) * attenuatedColor, 1.0 - ( 1.0 - transmittedLight.a ) * transmittanceFactor );
	}
#endif`,Y2=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_SPECULARMAP
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,q2=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	uniform mat3 mapTransform;
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	uniform mat3 alphaMapTransform;
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	uniform mat3 lightMapTransform;
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	uniform mat3 aoMapTransform;
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	uniform mat3 bumpMapTransform;
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	uniform mat3 normalMapTransform;
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_DISPLACEMENTMAP
	uniform mat3 displacementMapTransform;
	varying vec2 vDisplacementMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	uniform mat3 emissiveMapTransform;
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	uniform mat3 metalnessMapTransform;
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	uniform mat3 roughnessMapTransform;
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	uniform mat3 anisotropyMapTransform;
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	uniform mat3 clearcoatMapTransform;
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform mat3 clearcoatNormalMapTransform;
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform mat3 clearcoatRoughnessMapTransform;
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	uniform mat3 sheenColorMapTransform;
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	uniform mat3 sheenRoughnessMapTransform;
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	uniform mat3 iridescenceMapTransform;
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform mat3 iridescenceThicknessMapTransform;
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SPECULARMAP
	uniform mat3 specularMapTransform;
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	uniform mat3 specularColorMapTransform;
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	uniform mat3 specularIntensityMapTransform;
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,Q2=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	vUv = vec3( uv, 1 ).xy;
#endif
#ifdef USE_MAP
	vMapUv = ( mapTransform * vec3( MAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ALPHAMAP
	vAlphaMapUv = ( alphaMapTransform * vec3( ALPHAMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_LIGHTMAP
	vLightMapUv = ( lightMapTransform * vec3( LIGHTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_AOMAP
	vAoMapUv = ( aoMapTransform * vec3( AOMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_BUMPMAP
	vBumpMapUv = ( bumpMapTransform * vec3( BUMPMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_NORMALMAP
	vNormalMapUv = ( normalMapTransform * vec3( NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_DISPLACEMENTMAP
	vDisplacementMapUv = ( displacementMapTransform * vec3( DISPLACEMENTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_EMISSIVEMAP
	vEmissiveMapUv = ( emissiveMapTransform * vec3( EMISSIVEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_METALNESSMAP
	vMetalnessMapUv = ( metalnessMapTransform * vec3( METALNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ROUGHNESSMAP
	vRoughnessMapUv = ( roughnessMapTransform * vec3( ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ANISOTROPYMAP
	vAnisotropyMapUv = ( anisotropyMapTransform * vec3( ANISOTROPYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOATMAP
	vClearcoatMapUv = ( clearcoatMapTransform * vec3( CLEARCOATMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	vClearcoatNormalMapUv = ( clearcoatNormalMapTransform * vec3( CLEARCOAT_NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	vClearcoatRoughnessMapUv = ( clearcoatRoughnessMapTransform * vec3( CLEARCOAT_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCEMAP
	vIridescenceMapUv = ( iridescenceMapTransform * vec3( IRIDESCENCEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	vIridescenceThicknessMapUv = ( iridescenceThicknessMapTransform * vec3( IRIDESCENCE_THICKNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_COLORMAP
	vSheenColorMapUv = ( sheenColorMapTransform * vec3( SHEEN_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	vSheenRoughnessMapUv = ( sheenRoughnessMapTransform * vec3( SHEEN_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULARMAP
	vSpecularMapUv = ( specularMapTransform * vec3( SPECULARMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_COLORMAP
	vSpecularColorMapUv = ( specularColorMapTransform * vec3( SPECULAR_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	vSpecularIntensityMapUv = ( specularIntensityMapTransform * vec3( SPECULAR_INTENSITYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_TRANSMISSIONMAP
	vTransmissionMapUv = ( transmissionMapTransform * vec3( TRANSMISSIONMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_THICKNESSMAP
	vThicknessMapUv = ( thicknessMapTransform * vec3( THICKNESSMAP_UV, 1 ) ).xy;
#endif`,Z2=`#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`,K2=`varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`,J2=`uniform sampler2D t2D;
uniform float backgroundIntensity;
varying vec2 vUv;
void main() {
	vec4 texColor = texture2D( t2D, vUv );
	#ifdef DECODE_VIDEO_TEXTURE
		texColor = vec4( mix( pow( texColor.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), texColor.rgb * 0.0773993808, vec3( lessThanEqual( texColor.rgb, vec3( 0.04045 ) ) ) ), texColor.w );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,j2=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,$2=`#ifdef ENVMAP_TYPE_CUBE
	uniform samplerCube envMap;
#elif defined( ENVMAP_TYPE_CUBE_UV )
	uniform sampler2D envMap;
#endif
uniform float backgroundBlurriness;
uniform float backgroundIntensity;
uniform mat3 backgroundRotation;
varying vec3 vWorldDirection;
#include <cube_uv_reflection_fragment>
void main() {
	#ifdef ENVMAP_TYPE_CUBE
		vec4 texColor = textureCube( envMap, backgroundRotation * vWorldDirection );
	#elif defined( ENVMAP_TYPE_CUBE_UV )
		vec4 texColor = textureCubeUV( envMap, backgroundRotation * vWorldDirection, backgroundBlurriness );
	#else
		vec4 texColor = vec4( 0.0, 0.0, 0.0, 1.0 );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,eR=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,tR=`uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,nR=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
varying vec2 vHighPrecisionZW;
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vHighPrecisionZW = gl_Position.zw;
}`,iR=`#if DEPTH_PACKING == 3200
	uniform float opacity;
#endif
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
varying vec2 vHighPrecisionZW;
void main() {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#if DEPTH_PACKING == 3200
		diffuseColor.a = opacity;
	#endif
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <logdepthbuf_fragment>
	#ifdef USE_REVERSED_DEPTH_BUFFER
		float fragCoordZ = vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ];
	#else
		float fragCoordZ = 0.5 * vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ] + 0.5;
	#endif
	#if DEPTH_PACKING == 3200
		gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );
	#elif DEPTH_PACKING == 3201
		gl_FragColor = packDepthToRGBA( fragCoordZ );
	#elif DEPTH_PACKING == 3202
		gl_FragColor = vec4( packDepthToRGB( fragCoordZ ), 1.0 );
	#elif DEPTH_PACKING == 3203
		gl_FragColor = vec4( packDepthToRG( fragCoordZ ), 0.0, 1.0 );
	#endif
}`,sR=`#define DISTANCE
varying vec3 vWorldPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <worldpos_vertex>
	#include <clipping_planes_vertex>
	vWorldPosition = worldPosition.xyz;
}`,rR=`#define DISTANCE
uniform vec3 referencePosition;
uniform float nearDistance;
uniform float farDistance;
varying vec3 vWorldPosition;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	float dist = length( vWorldPosition - referencePosition );
	dist = ( dist - nearDistance ) / ( farDistance - nearDistance );
	dist = saturate( dist );
	gl_FragColor = vec4( dist, 0.0, 0.0, 1.0 );
}`,aR=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`,oR=`uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,lR=`uniform float scale;
attribute float lineDistance;
varying float vLineDistance;
#include <common>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	vLineDistance = scale * lineDistance;
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,cR=`uniform vec3 diffuse;
uniform float opacity;
uniform float dashSize;
uniform float totalSize;
varying float vLineDistance;
#include <common>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	if ( mod( vLineDistance, totalSize ) > dashSize ) {
		discard;
	}
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,uR=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#if defined ( USE_ENVMAP ) || defined ( USE_SKINNING )
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinbase_vertex>
		#include <skinnormal_vertex>
		#include <defaultnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <fog_vertex>
}`,fR=`uniform vec3 diffuse;
uniform float opacity;
#ifndef FLAT_SHADED
	varying vec3 vNormal;
#endif
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		reflectedLight.indirectDiffuse += lightMapTexel.rgb * lightMapIntensity * RECIPROCAL_PI;
	#else
		reflectedLight.indirectDiffuse += vec3( 1.0 );
	#endif
	#include <aomap_fragment>
	reflectedLight.indirectDiffuse *= diffuseColor.rgb;
	vec3 outgoingLight = reflectedLight.indirectDiffuse;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,hR=`#define LAMBERT
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,dR=`#define LAMBERT
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_lambert_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_lambert_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,pR=`#define MATCAP
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <displacementmap_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
	vViewPosition = - mvPosition.xyz;
}`,mR=`#define MATCAP
uniform vec3 diffuse;
uniform float opacity;
uniform sampler2D matcap;
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	vec3 viewDir = normalize( vViewPosition );
	vec3 x = normalize( vec3( viewDir.z, 0.0, - viewDir.x ) );
	vec3 y = cross( viewDir, x );
	vec2 uv = vec2( dot( x, normal ), dot( y, normal ) ) * 0.495 + 0.5;
	#ifdef USE_MATCAP
		vec4 matcapColor = texture2D( matcap, uv );
	#else
		vec4 matcapColor = vec4( vec3( mix( 0.2, 0.8, uv.y ) ), 1.0 );
	#endif
	vec3 outgoingLight = diffuseColor.rgb * matcapColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,gR=`#define NORMAL
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	vViewPosition = - mvPosition.xyz;
#endif
}`,vR=`#define NORMAL
uniform float opacity;
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <uv_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( 0.0, 0.0, 0.0, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	gl_FragColor = vec4( normalize( normal ) * 0.5 + 0.5, diffuseColor.a );
	#ifdef OPAQUE
		gl_FragColor.a = 1.0;
	#endif
}`,xR=`#define PHONG
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,yR=`#define PHONG
uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_phong_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_phong_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,_R=`#define STANDARD
varying vec3 vViewPosition;
#ifdef USE_TRANSMISSION
	varying vec3 vWorldPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
#ifdef USE_TRANSMISSION
	vWorldPosition = worldPosition.xyz;
#endif
}`,SR=`#define STANDARD
#ifdef PHYSICAL
	#define IOR
	#define USE_SPECULAR
#endif
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float roughness;
uniform float metalness;
uniform float opacity;
#ifdef IOR
	uniform float ior;
#endif
#ifdef USE_SPECULAR
	uniform float specularIntensity;
	uniform vec3 specularColor;
	#ifdef USE_SPECULAR_COLORMAP
		uniform sampler2D specularColorMap;
	#endif
	#ifdef USE_SPECULAR_INTENSITYMAP
		uniform sampler2D specularIntensityMap;
	#endif
#endif
#ifdef USE_CLEARCOAT
	uniform float clearcoat;
	uniform float clearcoatRoughness;
#endif
#ifdef USE_DISPERSION
	uniform float dispersion;
#endif
#ifdef USE_RETROREFLECTION
	uniform float retroreflectivity;
#endif
#ifdef USE_IRIDESCENCE
	uniform float iridescence;
	uniform float iridescenceIOR;
	uniform float iridescenceThicknessMinimum;
	uniform float iridescenceThicknessMaximum;
#endif
#ifdef USE_SHEEN
	uniform vec3 sheenColor;
	uniform float sheenRoughness;
	#ifdef USE_SHEEN_COLORMAP
		uniform sampler2D sheenColorMap;
	#endif
	#ifdef USE_SHEEN_ROUGHNESSMAP
		uniform sampler2D sheenRoughnessMap;
	#endif
#endif
#ifdef USE_ANISOTROPY
	uniform vec2 anisotropyVector;
	#ifdef USE_ANISOTROPYMAP
		uniform sampler2D anisotropyMap;
	#endif
#endif
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <iridescence_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_physical_pars_fragment>
#include <transmission_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <clearcoat_pars_fragment>
#include <iridescence_pars_fragment>
#include <roughnessmap_pars_fragment>
#include <metalnessmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <roughnessmap_fragment>
	#include <metalnessmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <clearcoat_normal_fragment_begin>
	#include <clearcoat_normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_physical_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 totalDiffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
	vec3 totalSpecular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;
	#include <transmission_fragment>
	vec3 outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance;
	#ifdef USE_SHEEN
 
		outgoingLight = outgoingLight + sheenSpecularDirect + sheenSpecularIndirect;
 
 	#endif
	#ifdef USE_CLEARCOAT
		float dotNVcc = saturate( dot( geometryClearcoatNormal, geometryViewDir ) );
		vec3 Fcc = F_Schlick( material.clearcoatF0, material.clearcoatF90, dotNVcc );
		outgoingLight = outgoingLight * ( 1.0 - material.clearcoat * Fcc ) + ( clearcoatSpecularDirect + clearcoatSpecularIndirect ) * material.clearcoat;
	#endif
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,AR=`#define TOON
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,MR=`#define TOON
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <gradientmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_toon_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_toon_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,ER=`uniform float size;
uniform float scale;
#include <common>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
#ifdef USE_POINTS_UV
	varying vec2 vUv;
	uniform mat3 uvTransform;
#endif
void main() {
	#ifdef USE_POINTS_UV
		vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	#endif
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	gl_PointSize = size;
	#ifdef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) gl_PointSize *= ( scale / - mvPosition.z );
	#endif
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <fog_vertex>
}`,TR=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <color_pars_fragment>
#include <map_particle_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_particle_fragment>
	#include <color_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,bR=`#include <common>
#include <batching_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <shadowmap_pars_vertex>
void main() {
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,wR=`uniform vec3 color;
uniform float opacity;
#include <common>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <logdepthbuf_pars_fragment>
#include <shadowmap_pars_fragment>
#include <shadowmask_pars_fragment>
void main() {
	#include <logdepthbuf_fragment>
	gl_FragColor = vec4( color, opacity * ( 1.0 - getShadowMask() ) );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,CR=`uniform float rotation;
uniform vec2 center;
#include <common>
#include <uv_pars_vertex>
#include <fog_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	vec4 mvPosition = modelViewMatrix[ 3 ];
	vec2 scale = vec2( length( modelMatrix[ 0 ].xyz ), length( modelMatrix[ 1 ].xyz ) );
	#ifndef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) scale *= - mvPosition.z;
	#endif
	vec2 alignedPosition = ( position.xy - ( center - vec2( 0.5 ) ) ) * scale;
	vec2 rotatedPosition;
	rotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;
	rotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;
	mvPosition.xy += rotatedPosition;
	gl_Position = projectionMatrix * mvPosition;
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,RR=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`,He={alphahash_fragment:Kw,alphahash_pars_fragment:Jw,alphamap_fragment:jw,alphamap_pars_fragment:$w,alphatest_fragment:eC,alphatest_pars_fragment:tC,aomap_fragment:nC,aomap_pars_fragment:iC,batching_pars_vertex:sC,batching_vertex:rC,begin_vertex:aC,beginnormal_vertex:oC,bsdfs:lC,iridescence_fragment:cC,bumpmap_pars_fragment:uC,clipping_planes_fragment:fC,clipping_planes_pars_fragment:hC,clipping_planes_pars_vertex:dC,clipping_planes_vertex:pC,color_fragment:mC,color_pars_fragment:gC,color_pars_vertex:vC,color_vertex:xC,common:yC,cube_uv_reflection_fragment:_C,defaultnormal_vertex:SC,displacementmap_pars_vertex:AC,displacementmap_vertex:MC,emissivemap_fragment:EC,emissivemap_pars_fragment:TC,colorspace_fragment:bC,colorspace_pars_fragment:wC,envmap_fragment:CC,envmap_common_pars_fragment:RC,envmap_pars_fragment:DC,envmap_pars_vertex:UC,envmap_physical_pars_fragment:VC,envmap_vertex:IC,fog_vertex:BC,fog_pars_vertex:NC,fog_fragment:PC,fog_pars_fragment:LC,gradientmap_pars_fragment:OC,lightmap_pars_fragment:FC,lights_lambert_fragment:zC,lights_lambert_pars_fragment:HC,lights_pars_begin:GC,lights_toon_fragment:kC,lights_toon_pars_fragment:WC,lights_phong_fragment:XC,lights_phong_pars_fragment:YC,lights_physical_fragment:qC,lights_physical_pars_fragment:QC,lights_fragment_begin:ZC,lights_fragment_maps:KC,lights_fragment_end:JC,lightprobes_pars_fragment:jC,logdepthbuf_fragment:$C,logdepthbuf_pars_fragment:e2,logdepthbuf_pars_vertex:t2,logdepthbuf_vertex:n2,map_fragment:i2,map_pars_fragment:s2,map_particle_fragment:r2,map_particle_pars_fragment:a2,metalnessmap_fragment:o2,metalnessmap_pars_fragment:l2,morphinstance_vertex:c2,morphcolor_vertex:u2,morphnormal_vertex:f2,morphtarget_pars_vertex:h2,morphtarget_vertex:d2,normal_fragment_begin:p2,normal_fragment_maps:m2,normal_pars_fragment:g2,normal_pars_vertex:v2,normal_vertex:x2,normalmap_pars_fragment:y2,clearcoat_normal_fragment_begin:_2,clearcoat_normal_fragment_maps:S2,clearcoat_pars_fragment:A2,iridescence_pars_fragment:M2,opaque_fragment:E2,packing:T2,premultiplied_alpha_fragment:b2,project_vertex:w2,dithering_fragment:C2,dithering_pars_fragment:R2,roughnessmap_fragment:D2,roughnessmap_pars_fragment:U2,shadowmap_pars_fragment:I2,shadowmap_pars_vertex:B2,shadowmap_vertex:N2,shadowmask_pars_fragment:P2,skinbase_vertex:L2,skinning_pars_vertex:O2,skinning_vertex:F2,skinnormal_vertex:z2,specularmap_fragment:H2,specularmap_pars_fragment:G2,tonemapping_fragment:V2,tonemapping_pars_fragment:k2,transmission_fragment:W2,transmission_pars_fragment:X2,uv_pars_fragment:Y2,uv_pars_vertex:q2,uv_vertex:Q2,worldpos_vertex:Z2,background_vert:K2,background_frag:J2,backgroundCube_vert:j2,backgroundCube_frag:$2,cube_vert:eR,cube_frag:tR,depth_vert:nR,depth_frag:iR,distance_vert:sR,distance_frag:rR,equirect_vert:aR,equirect_frag:oR,linedashed_vert:lR,linedashed_frag:cR,meshbasic_vert:uR,meshbasic_frag:fR,meshlambert_vert:hR,meshlambert_frag:dR,meshmatcap_vert:pR,meshmatcap_frag:mR,meshnormal_vert:gR,meshnormal_frag:vR,meshphong_vert:xR,meshphong_frag:yR,meshphysical_vert:_R,meshphysical_frag:SR,meshtoon_vert:AR,meshtoon_frag:MR,points_vert:ER,points_frag:TR,shadow_vert:bR,shadow_frag:wR,sprite_vert:CR,sprite_frag:RR},he={common:{diffuse:{value:new Ye(16777215)},opacity:{value:1},map:{value:null},mapTransform:{value:new Pe},alphaMap:{value:null},alphaMapTransform:{value:new Pe},alphaTest:{value:0}},specularmap:{specularMap:{value:null},specularMapTransform:{value:new Pe}},envmap:{envMap:{value:null},envMapRotation:{value:new Pe},reflectivity:{value:1},ior:{value:1.5},refractionRatio:{value:.98},dfgLUT:{value:null}},aomap:{aoMap:{value:null},aoMapIntensity:{value:1},aoMapTransform:{value:new Pe}},lightmap:{lightMap:{value:null},lightMapIntensity:{value:1},lightMapTransform:{value:new Pe}},bumpmap:{bumpMap:{value:null},bumpMapTransform:{value:new Pe},bumpScale:{value:1}},normalmap:{normalMap:{value:null},normalMapTransform:{value:new Pe},normalScale:{value:new Ne(1,1)}},displacementmap:{displacementMap:{value:null},displacementMapTransform:{value:new Pe},displacementScale:{value:1},displacementBias:{value:0}},emissivemap:{emissiveMap:{value:null},emissiveMapTransform:{value:new Pe}},metalnessmap:{metalnessMap:{value:null},metalnessMapTransform:{value:new Pe}},roughnessmap:{roughnessMap:{value:null},roughnessMapTransform:{value:new Pe}},gradientmap:{gradientMap:{value:null}},fog:{fogDensity:{value:25e-5},fogNear:{value:1},fogFar:{value:2e3},fogColor:{value:new Ye(16777215)}},lights:{ambientLightColor:{value:[]},lightProbe:{value:[]},sunLights:{value:[],properties:{direction:{},color:{}}},sunLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},sunShadowMatrix:{value:[]},sunShadowCascade:{value:[]},directionalLights:{value:[],properties:{direction:{},color:{}}},directionalLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},directionalShadowMatrix:{value:[]},spotLights:{value:[],properties:{color:{},position:{},direction:{},distance:{},coneCos:{},penumbraCos:{},decay:{}}},spotLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},spotLightMap:{value:[]},spotLightMatrix:{value:[]},pointLights:{value:[],properties:{color:{},position:{},decay:{},distance:{}}},pointLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{},shadowCameraNear:{},shadowCameraFar:{}}},pointShadowMatrix:{value:[]},hemisphereLights:{value:[],properties:{direction:{},skyColor:{},groundColor:{}}},rectAreaLights:{value:[],properties:{color:{},position:{},width:{},height:{}}},ltc_1:{value:null},ltc_2:{value:null},probesSH:{value:null},probesMin:{value:new G},probesMax:{value:new G},probesResolution:{value:new G}},points:{diffuse:{value:new Ye(16777215)},opacity:{value:1},size:{value:1},scale:{value:1},map:{value:null},alphaMap:{value:null},alphaMapTransform:{value:new Pe},alphaTest:{value:0},uvTransform:{value:new Pe}},sprite:{diffuse:{value:new Ye(16777215)},opacity:{value:1},center:{value:new Ne(.5,.5)},rotation:{value:0},map:{value:null},mapTransform:{value:new Pe},alphaMap:{value:null},alphaMapTransform:{value:new Pe},alphaTest:{value:0}}},xs={basic:{uniforms:Dn([he.common,he.specularmap,he.envmap,he.aomap,he.lightmap,he.fog]),vertexShader:He.meshbasic_vert,fragmentShader:He.meshbasic_frag},lambert:{uniforms:Dn([he.common,he.specularmap,he.envmap,he.aomap,he.lightmap,he.emissivemap,he.bumpmap,he.normalmap,he.displacementmap,he.fog,he.lights,{emissive:{value:new Ye(0)},envMapIntensity:{value:1}}]),vertexShader:He.meshlambert_vert,fragmentShader:He.meshlambert_frag},phong:{uniforms:Dn([he.common,he.specularmap,he.envmap,he.aomap,he.lightmap,he.emissivemap,he.bumpmap,he.normalmap,he.displacementmap,he.fog,he.lights,{emissive:{value:new Ye(0)},specular:{value:new Ye(1118481)},shininess:{value:30},envMapIntensity:{value:1}}]),vertexShader:He.meshphong_vert,fragmentShader:He.meshphong_frag},standard:{uniforms:Dn([he.common,he.envmap,he.aomap,he.lightmap,he.emissivemap,he.bumpmap,he.normalmap,he.displacementmap,he.roughnessmap,he.metalnessmap,he.fog,he.lights,{emissive:{value:new Ye(0)},roughness:{value:1},metalness:{value:0},envMapIntensity:{value:1}}]),vertexShader:He.meshphysical_vert,fragmentShader:He.meshphysical_frag},toon:{uniforms:Dn([he.common,he.aomap,he.lightmap,he.emissivemap,he.bumpmap,he.normalmap,he.displacementmap,he.gradientmap,he.fog,he.lights,{emissive:{value:new Ye(0)}}]),vertexShader:He.meshtoon_vert,fragmentShader:He.meshtoon_frag},matcap:{uniforms:Dn([he.common,he.bumpmap,he.normalmap,he.displacementmap,he.fog,{matcap:{value:null}}]),vertexShader:He.meshmatcap_vert,fragmentShader:He.meshmatcap_frag},points:{uniforms:Dn([he.points,he.fog]),vertexShader:He.points_vert,fragmentShader:He.points_frag},dashed:{uniforms:Dn([he.common,he.fog,{scale:{value:1},dashSize:{value:1},totalSize:{value:2}}]),vertexShader:He.linedashed_vert,fragmentShader:He.linedashed_frag},depth:{uniforms:Dn([he.common,he.displacementmap]),vertexShader:He.depth_vert,fragmentShader:He.depth_frag},normal:{uniforms:Dn([he.common,he.bumpmap,he.normalmap,he.displacementmap,{opacity:{value:1}}]),vertexShader:He.meshnormal_vert,fragmentShader:He.meshnormal_frag},sprite:{uniforms:Dn([he.sprite,he.fog]),vertexShader:He.sprite_vert,fragmentShader:He.sprite_frag},background:{uniforms:{uvTransform:{value:new Pe},t2D:{value:null},backgroundIntensity:{value:1}},vertexShader:He.background_vert,fragmentShader:He.background_frag},backgroundCube:{uniforms:{envMap:{value:null},backgroundBlurriness:{value:0},backgroundIntensity:{value:1},backgroundRotation:{value:new Pe}},vertexShader:He.backgroundCube_vert,fragmentShader:He.backgroundCube_frag},cube:{uniforms:{tCube:{value:null},tFlip:{value:-1},opacity:{value:1}},vertexShader:He.cube_vert,fragmentShader:He.cube_frag},equirect:{uniforms:{tEquirect:{value:null}},vertexShader:He.equirect_vert,fragmentShader:He.equirect_frag},distance:{uniforms:Dn([he.common,he.displacementmap,{referencePosition:{value:new G},nearDistance:{value:1},farDistance:{value:1e3}}]),vertexShader:He.distance_vert,fragmentShader:He.distance_frag},shadow:{uniforms:Dn([he.lights,he.fog,{color:{value:new Ye(0)},opacity:{value:1}}]),vertexShader:He.shadow_vert,fragmentShader:He.shadow_frag}};xs.physical={uniforms:Dn([xs.standard.uniforms,{clearcoat:{value:0},clearcoatMap:{value:null},clearcoatMapTransform:{value:new Pe},clearcoatNormalMap:{value:null},clearcoatNormalMapTransform:{value:new Pe},clearcoatNormalScale:{value:new Ne(1,1)},clearcoatRoughness:{value:0},clearcoatRoughnessMap:{value:null},clearcoatRoughnessMapTransform:{value:new Pe},dispersion:{value:0},retroreflectivity:{value:0},iridescence:{value:0},iridescenceMap:{value:null},iridescenceMapTransform:{value:new Pe},iridescenceIOR:{value:1.3},iridescenceThicknessMinimum:{value:100},iridescenceThicknessMaximum:{value:400},iridescenceThicknessMap:{value:null},iridescenceThicknessMapTransform:{value:new Pe},sheen:{value:0},sheenColor:{value:new Ye(0)},sheenColorMap:{value:null},sheenColorMapTransform:{value:new Pe},sheenRoughness:{value:1},sheenRoughnessMap:{value:null},sheenRoughnessMapTransform:{value:new Pe},transmission:{value:0},transmissionMap:{value:null},transmissionMapTransform:{value:new Pe},transmissionSamplerSize:{value:new Ne},transmissionSamplerMap:{value:null},thickness:{value:0},thicknessMap:{value:null},thicknessMapTransform:{value:new Pe},attenuationDistance:{value:0},attenuationColor:{value:new Ye(0)},specularColor:{value:new Ye(1,1,1)},specularColorMap:{value:null},specularColorMapTransform:{value:new Pe},specularIntensity:{value:1},specularIntensityMap:{value:null},specularIntensityMapTransform:{value:new Pe},anisotropyVector:{value:new Ne},anisotropyMap:{value:null},anisotropyMapTransform:{value:new Pe}}]),vertexShader:He.meshphysical_vert,fragmentShader:He.meshphysical_frag};var pd={r:0,b:0,g:0},DR=new kt,fM=new Pe;fM.set(-1,0,0,0,1,0,0,0,1);function UR(t,e,n,i,s,r){let a=new Ye(0),o=s===!0?0:1,l,c,h=null,p=0,u=null;function d(g){let S=g.isScene===!0?g.background:null;if(S&&S.isTexture){let _=g.backgroundBlurriness>0;S=e.get(S,_)}return S}function v(g){let S=!1,_=d(g);_===null?m(a,o):_&&_.isColor&&(m(_,1),S=!0);let E=t.xr.getEnvironmentBlendMode();E==="additive"?n.buffers.color.setClear(0,0,0,1,r):E==="alpha-blend"&&n.buffers.color.setClear(0,0,0,0,r),(t.autoClear||S)&&(n.buffers.depth.setTest(!0),n.buffers.depth.setMask(!0),n.buffers.color.setMask(!0),t.clear(t.autoClearColor,t.autoClearDepth,t.autoClearStencil))}function M(g,S){let _=d(S);_&&(_.isCubeTexture||_.mapping===Tc)?(c===void 0&&(c=new Rn(new qo(1,1,1),new Wt({name:"BackgroundCubeMaterial",uniforms:Aa(xs.backgroundCube.uniforms),vertexShader:xs.backgroundCube.vertexShader,fragmentShader:xs.backgroundCube.fragmentShader,side:$t,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),c.geometry.deleteAttribute("normal"),c.geometry.deleteAttribute("uv"),c.onBeforeRender=function(E,T,C){this.matrixWorld.copyPosition(C.matrixWorld)},Object.defineProperty(c.material,"envMap",{get:function(){return this.uniforms.envMap.value}}),i.update(c)),c.material.uniforms.envMap.value=_,c.material.uniforms.backgroundBlurriness.value=S.backgroundBlurriness,c.material.uniforms.backgroundIntensity.value=S.backgroundIntensity,c.material.uniforms.backgroundRotation.value.setFromMatrix4(DR.makeRotationFromEuler(S.backgroundRotation)).transpose(),_.isCubeTexture&&_.isRenderTargetTexture===!1&&c.material.uniforms.backgroundRotation.value.premultiply(fM),c.material.toneMapped=Ke.getTransfer(_.colorSpace)!==ct,(h!==_||p!==_.version||u!==t.toneMapping)&&(c.material.needsUpdate=!0,h=_,p=_.version,u=t.toneMapping),c.layers.enableAll(),g.unshift(c,c.geometry,c.material,0,0,null)):_&&_.isTexture&&(l===void 0&&(l=new Rn(new ya(2,2),new Wt({name:"BackgroundMaterial",uniforms:Aa(xs.background.uniforms),vertexShader:xs.background.vertexShader,fragmentShader:xs.background.fragmentShader,side:ps,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),l.geometry.deleteAttribute("normal"),Object.defineProperty(l.material,"map",{get:function(){return this.uniforms.t2D.value}}),i.update(l)),l.material.uniforms.t2D.value=_,l.material.uniforms.backgroundIntensity.value=S.backgroundIntensity,l.material.toneMapped=Ke.getTransfer(_.colorSpace)!==ct,_.matrixAutoUpdate===!0&&_.updateMatrix(),l.material.uniforms.uvTransform.value.copy(_.matrix),(h!==_||p!==_.version||u!==t.toneMapping)&&(l.material.needsUpdate=!0,h=_,p=_.version,u=t.toneMapping),l.layers.enableAll(),g.unshift(l,l.geometry,l.material,0,0,null))}function m(g,S){g.getRGB(pd,C0(t)),n.buffers.color.setClear(pd.r,pd.g,pd.b,S,r)}function f(){c!==void 0&&(c.geometry.dispose(),c.material.dispose(),c=void 0),l!==void 0&&(l.geometry.dispose(),l.material.dispose(),l=void 0)}return{getClearColor:function(){return a},setClearColor:function(g,S=1){a.set(g),o=S,m(a,o)},getClearAlpha:function(){return o},setClearAlpha:function(g){o=g,m(a,o)},render:v,addToRenderList:M,dispose:f}}function IR(t,e){let n=t.getParameter(t.MAX_VERTEX_ATTRIBS),i={},s=u(null),r=s,a=!1;function o(N,F,k,B,z){let Z=!1,q=p(N,B,k,F);r!==q&&(r=q,c(r.object)),Z=d(N,B,k,z),Z&&v(N,B,k,z),z!==null&&e.update(z,t.ELEMENT_ARRAY_BUFFER),(Z||a)&&(a=!1,_(N,F,k,B),z!==null&&t.bindBuffer(t.ELEMENT_ARRAY_BUFFER,e.get(z).buffer))}function l(){return t.createVertexArray()}function c(N){return t.bindVertexArray(N)}function h(N){return t.deleteVertexArray(N)}function p(N,F,k,B){let z=B.wireframe===!0,Z=i[F.id];Z===void 0&&(Z={},i[F.id]=Z);let q=N.isInstancedMesh===!0?N.id:0,ie=Z[q];ie===void 0&&(ie={},Z[q]=ie);let W=ie[k.id];W===void 0&&(W={},ie[k.id]=W);let $=W[z];return $===void 0&&($=u(l()),W[z]=$),$}function u(N){let F=[],k=[],B=[];for(let z=0;z<n;z++)F[z]=0,k[z]=0,B[z]=0;return{geometry:null,program:null,wireframe:!1,newAttributes:F,enabledAttributes:k,attributeDivisors:B,object:N,attributes:{},index:null}}function d(N,F,k,B){let z=r.attributes,Z=F.attributes,q=0,ie=k.getAttributes();for(let W in ie)if(ie[W].location>=0){let te=z[W],we=Z[W];if(we===void 0&&(W==="instanceMatrix"&&N.instanceMatrix&&(we=N.instanceMatrix),W==="instanceColor"&&N.instanceColor&&(we=N.instanceColor)),te===void 0||te.attribute!==we||we&&te.data!==we.data)return!0;q++}return r.attributesNum!==q||r.index!==B}function v(N,F,k,B){let z={},Z=F.attributes,q=0,ie=k.getAttributes();for(let W in ie)if(ie[W].location>=0){let te=Z[W];te===void 0&&(W==="instanceMatrix"&&N.instanceMatrix&&(te=N.instanceMatrix),W==="instanceColor"&&N.instanceColor&&(te=N.instanceColor));let we={};we.attribute=te,te&&te.data&&(we.data=te.data),z[W]=we,q++}r.attributes=z,r.attributesNum=q,r.index=B}function M(){let N=r.newAttributes;for(let F=0,k=N.length;F<k;F++)N[F]=0}function m(N){f(N,0)}function f(N,F){let k=r.newAttributes,B=r.enabledAttributes,z=r.attributeDivisors;k[N]=1,B[N]===0&&(t.enableVertexAttribArray(N),B[N]=1),z[N]!==F&&(t.vertexAttribDivisor(N,F),z[N]=F)}function g(){let N=r.newAttributes,F=r.enabledAttributes;for(let k=0,B=F.length;k<B;k++)F[k]!==N[k]&&(t.disableVertexAttribArray(k),F[k]=0)}function S(N,F,k,B,z,Z,q){q===!0?t.vertexAttribIPointer(N,F,k,z,Z):t.vertexAttribPointer(N,F,k,B,z,Z)}function _(N,F,k,B){M();let z=B.attributes,Z=k.getAttributes(),q=F.defaultAttributeValues;for(let ie in Z){let W=Z[ie];if(W.location>=0){let $=z[ie];if($===void 0&&(ie==="instanceMatrix"&&N.instanceMatrix&&($=N.instanceMatrix),ie==="instanceColor"&&N.instanceColor&&($=N.instanceColor)),$!==void 0){let te=$.normalized,we=$.itemSize,Me=e.get($);if(Me===void 0)continue;let ut=Me.buffer,qe=Me.type,$e=Me.bytesPerElement,X=qe===t.INT||qe===t.UNSIGNED_INT||$.gpuType===Dh;if($.isInterleavedBufferAttribute){let ee=$.data,ue=ee.stride,Se=$.offset;if(ee.isInstancedInterleavedBuffer){for(let ge=0;ge<W.locationSize;ge++)f(W.location+ge,ee.meshPerAttribute);N.isInstancedMesh!==!0&&B._maxInstanceCount===void 0&&(B._maxInstanceCount=ee.meshPerAttribute*ee.count)}else for(let ge=0;ge<W.locationSize;ge++)m(W.location+ge);t.bindBuffer(t.ARRAY_BUFFER,ut);for(let ge=0;ge<W.locationSize;ge++)S(W.location+ge,we/W.locationSize,qe,te,ue*$e,(Se+we/W.locationSize*ge)*$e,X)}else{if($.isInstancedBufferAttribute){for(let ee=0;ee<W.locationSize;ee++)f(W.location+ee,$.meshPerAttribute);N.isInstancedMesh!==!0&&B._maxInstanceCount===void 0&&(B._maxInstanceCount=$.meshPerAttribute*$.count)}else for(let ee=0;ee<W.locationSize;ee++)m(W.location+ee);t.bindBuffer(t.ARRAY_BUFFER,ut);for(let ee=0;ee<W.locationSize;ee++)S(W.location+ee,we/W.locationSize,qe,te,we*$e,we/W.locationSize*ee*$e,X)}}else if(q!==void 0){let te=q[ie];if(te!==void 0)switch(te.length){case 2:t.vertexAttrib2fv(W.location,te);break;case 3:t.vertexAttrib3fv(W.location,te);break;case 4:t.vertexAttrib4fv(W.location,te);break;default:t.vertexAttrib1fv(W.location,te)}}}}g()}function E(){b();for(let N in i){let F=i[N];for(let k in F){let B=F[k];for(let z in B){let Z=B[z];for(let q in Z)h(Z[q].object),delete Z[q];delete B[z]}}delete i[N]}}function T(N){if(i[N.id]===void 0)return;let F=i[N.id];for(let k in F){let B=F[k];for(let z in B){let Z=B[z];for(let q in Z)h(Z[q].object),delete Z[q];delete B[z]}}delete i[N.id]}function C(N){for(let F in i){let k=i[F];for(let B in k){let z=k[B];if(z[N.id]===void 0)continue;let Z=z[N.id];for(let q in Z)h(Z[q].object),delete Z[q];delete z[N.id]}}}function y(N){for(let F in i){let k=i[F],B=N.isInstancedMesh===!0?N.id:0,z=k[B];if(z!==void 0){for(let Z in z){let q=z[Z];for(let ie in q)h(q[ie].object),delete q[ie];delete z[Z]}delete k[B],Object.keys(k).length===0&&delete i[F]}}}function b(){R(),a=!0,r!==s&&(r=s,c(r.object))}function R(){s.geometry=null,s.program=null,s.wireframe=!1}return{setup:o,reset:b,resetDefaultState:R,dispose:E,releaseStatesOfGeometry:T,releaseStatesOfObject:y,releaseStatesOfProgram:C,initAttributes:M,enableAttribute:m,disableUnusedAttributes:g}}function BR(t,e,n){let i;function s(l){i=l}function r(l,c){t.drawArrays(i,l,c),n.update(c,i,1)}function a(l,c,h){h!==0&&(t.drawArraysInstanced(i,l,c,h),n.update(c,i,h))}function o(l,c,h){if(h===0)return;e.get("WEBGL_multi_draw").multiDrawArraysWEBGL(i,l,0,c,0,h);let u=0;for(let d=0;d<h;d++)u+=c[d];n.update(u,i,1)}this.setMode=s,this.render=r,this.renderInstances=a,this.renderMultiDraw=o}function NR(t,e,n,i){let s;function r(){if(s!==void 0)return s;if(e.has("EXT_texture_filter_anisotropic")===!0){let C=e.get("EXT_texture_filter_anisotropic");s=t.getParameter(C.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else s=0;return s}function a(C){return!(C!==Ri&&i.convert(C)!==t.getParameter(t.IMPLEMENTATION_COLOR_READ_FORMAT))}function o(C){let y=C===Xi&&(e.has("EXT_color_buffer_half_float")||e.has("EXT_color_buffer_float"));return!(C!==Xt&&C!==fi&&!y&&i.convert(C)!==t.getParameter(t.IMPLEMENTATION_COLOR_READ_TYPE))}function l(C){if(C==="highp"){if(t.getShaderPrecisionFormat(t.VERTEX_SHADER,t.HIGH_FLOAT).precision>0&&t.getShaderPrecisionFormat(t.FRAGMENT_SHADER,t.HIGH_FLOAT).precision>0)return"highp";C="mediump"}return C==="mediump"&&t.getShaderPrecisionFormat(t.VERTEX_SHADER,t.MEDIUM_FLOAT).precision>0&&t.getShaderPrecisionFormat(t.FRAGMENT_SHADER,t.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}let c=n.precision!==void 0?n.precision:"highp",h=l(c);h!==c&&(De("WebGLRenderer:",c,"not supported, using",h,"instead."),c=h);let p=n.logarithmicDepthBuffer===!0,u=n.reversedDepthBuffer===!0&&e.has("EXT_clip_control");n.reversedDepthBuffer===!0&&u===!1&&De("WebGLRenderer: Unable to use reversed depth buffer due to missing EXT_clip_control extension. Fallback to default depth buffer.");let d=t.getParameter(t.MAX_TEXTURE_IMAGE_UNITS),v=t.getParameter(t.MAX_VERTEX_TEXTURE_IMAGE_UNITS),M=t.getParameter(t.MAX_TEXTURE_SIZE),m=t.getParameter(t.MAX_CUBE_MAP_TEXTURE_SIZE),f=t.getParameter(t.MAX_VERTEX_ATTRIBS),g=t.getParameter(t.MAX_VERTEX_UNIFORM_VECTORS),S=t.getParameter(t.MAX_VARYING_VECTORS),_=t.getParameter(t.MAX_FRAGMENT_UNIFORM_VECTORS),E=t.getParameter(t.MAX_SAMPLES),T=t.getParameter(t.SAMPLES);return{isWebGL2:!0,getMaxAnisotropy:r,getMaxPrecision:l,textureFormatReadable:a,textureTypeReadable:o,precision:c,logarithmicDepthBuffer:p,reversedDepthBuffer:u,maxTextures:d,maxVertexTextures:v,maxTextureSize:M,maxCubemapSize:m,maxAttributes:f,maxVertexUniforms:g,maxVaryings:S,maxFragmentUniforms:_,maxSamples:E,samples:T}}function PR(t){let e=this,n=null,i=0,s=!1,r=!1,a=new Hi,o=new Pe,l={value:null,needsUpdate:!1};this.uniform=l,this.numPlanes=0,this.numIntersection=0,this.init=function(p,u){let d=p.length!==0||u||i!==0||s;return s=u,i=p.length,d},this.beginShadows=function(){r=!0,h(null)},this.endShadows=function(){r=!1},this.setGlobalState=function(p,u){n=h(p,u,0)},this.setState=function(p,u,d){let v=p.clippingPlanes,M=p.clipIntersection,m=p.clipShadows,f=t.get(p);if(!s||v===null||v.length===0||r&&!m)r?h(null):c();else{let g=r?0:i,S=g*4,_=f.clippingState||null;l.value=_,_=h(v,u,S,d);for(let E=0;E!==S;++E)_[E]=n[E];f.clippingState=_,this.numIntersection=M?this.numPlanes:0,this.numPlanes+=g}};function c(){l.value!==n&&(l.value=n,l.needsUpdate=i>0),e.numPlanes=i,e.numIntersection=0}function h(p,u,d,v){let M=p!==null?p.length:0,m=null;if(M!==0){if(m=l.value,v!==!0||m===null){let f=d+M*4,g=u.matrixWorldInverse;o.getNormalMatrix(g),(m===null||m.length<f)&&(m=new Float32Array(f));for(let S=0,_=d;S!==M;++S,_+=4)a.copy(p[S]).applyMatrix4(g,o),a.normal.toArray(m,_),m[_+3]=a.constant}l.value=m,l.needsUpdate=!0}return e.numPlanes=M,e.numIntersection=0,m}}var jo=4,LR=6,OR=20,FR=256,Pc=new Vs,k1=new Ye,O0=null,F0=0,z0=0,H0=!1,zR=new G,Ma=new G,gd=class{constructor(e){this._renderer=e,this._pingPongRenderTarget=null,this._lodMax=0,this._cubeSize=0,this._sizeLods=[],this._lodMeshes=[],this._backgroundBox=null,this._cubemapMaterial=null,this._equirectMaterial=null,this._blurMaterial=null,this._ggxMaterial=null}fromScene(e,n=0,i=.1,s=100,r={}){let{size:a=256,position:o=zR}=r;O0=this._renderer.getRenderTarget(),F0=this._renderer.getActiveCubeFace(),z0=this._renderer.getActiveMipmapLevel(),H0=this._renderer.xr.enabled,this._renderer.xr.enabled=!1,this._setSize(a);let l=this._allocateTargets();return l.depthBuffer=!0,this._sceneToCubeUV(e,i,s,l,o),n>0&&this._blur(l,0,0,n),this._applyPMREM(l),this._cleanup(l),l}fromEquirectangular(e,n=null){return this._fromTexture(e,n)}fromCubemap(e,n=null){return this._fromTexture(e,n)}compileCubemapShader(){this._cubemapMaterial===null&&(this._cubemapMaterial=Y1(),this._compileMaterial(this._cubemapMaterial))}compileEquirectangularShader(){this._equirectMaterial===null&&(this._equirectMaterial=X1(),this._compileMaterial(this._equirectMaterial))}dispose(){this._dispose(),this._cubemapMaterial!==null&&this._cubemapMaterial.dispose(),this._equirectMaterial!==null&&this._equirectMaterial.dispose(),this._backgroundBox!==null&&(this._backgroundBox.geometry.dispose(),this._backgroundBox.material.dispose())}_setSize(e){this._lodMax=Math.floor(Math.log2(e)),this._cubeSize=Math.pow(2,this._lodMax)}_dispose(){this._blurMaterial!==null&&this._blurMaterial.dispose(),this._ggxMaterial!==null&&this._ggxMaterial.dispose(),this._pingPongRenderTarget!==null&&this._pingPongRenderTarget.dispose();for(let e=0;e<this._lodMeshes.length;e++)this._lodMeshes[e].geometry.dispose()}_cleanup(e){this._renderer.setRenderTarget(O0,F0,z0),this._renderer.xr.enabled=H0,e.scissorTest=!1,Jo(e,0,0,e.width,e.height)}_fromTexture(e,n){e.mapping===Pr||e.mapping===Sa?this._setSize(e.image.length===0?16:e.image[0].width||e.image[0].image.width):this._setSize(e.image.width/4),O0=this._renderer.getRenderTarget(),F0=this._renderer.getActiveCubeFace(),z0=this._renderer.getActiveMipmapLevel(),H0=this._renderer.xr.enabled,this._renderer.xr.enabled=!1;let i=n||this._allocateTargets();return this._textureToCubeUV(e,i),this._applyPMREM(i),this._cleanup(i),i}_allocateTargets(){let e=3*Math.max(this._cubeSize,112),n=4*this._cubeSize,i={magFilter:wt,minFilter:wt,generateMipmaps:!1,type:Xi,format:Ri,colorSpace:Gs,depthBuffer:!1},s=W1(e,n,i);if(this._pingPongRenderTarget===null||this._pingPongRenderTarget.width!==e||this._pingPongRenderTarget.height!==n){this._pingPongRenderTarget!==null&&this._dispose(),this._pingPongRenderTarget=W1(e,n,i);let{_lodMax:r}=this;({lodMeshes:this._lodMeshes,sizeLods:this._sizeLods}=HR(r)),this._blurMaterial=VR(r,e,n),this._ggxMaterial=GR(r,e,n)}return s}_compileMaterial(e){let n=new Rn(new wi,e);this._renderer.compile(n,Pc)}_sceneToCubeUV(e,n,i,s,r){let l=new Cn(90,1,n,i),c=[1,-1,1,1,1,1],h=[1,1,1,-1,-1,-1],p=this._renderer,u=p.autoClear,d=p.toneMapping;p.getClearColor(k1),p.toneMapping=ki,p.autoClear=!1,p.state.buffers.depth.getReversed()&&(p.setRenderTarget(s),p.clearDepth(),p.setRenderTarget(null)),this._backgroundBox===null&&(this._backgroundBox=new Rn(new qo,new vc({name:"PMREM.Background",side:$t,depthWrite:!1,depthTest:!1})));let M=this._backgroundBox,m=M.material,f=!1,g=e.background;g?g.isColor&&(m.color.copy(g),e.background=null,f=!0):(m.color.copy(k1),f=!0);for(let S=0;S<6;S++){let _=S%3;_===0?(l.up.set(0,c[S],0),l.position.set(r.x,r.y,r.z),l.lookAt(r.x+h[S],r.y,r.z)):_===1?(l.up.set(0,0,c[S]),l.position.set(r.x,r.y,r.z),l.lookAt(r.x,r.y+h[S],r.z)):(l.up.set(0,c[S],0),l.position.set(r.x,r.y,r.z),l.lookAt(r.x,r.y,r.z+h[S]));let E=this._cubeSize;Jo(s,_*E,S>2?E:0,E,E),p.setRenderTarget(s),f&&p.render(M,l),p.render(e,l)}p.toneMapping=d,p.autoClear=u,e.background=g}_textureToCubeUV(e,n){let i=this._renderer,s=e.mapping===Pr||e.mapping===Sa;s?(this._cubemapMaterial===null&&(this._cubemapMaterial=Y1()),this._cubemapMaterial.uniforms.flipEnvMap.value=e.isRenderTargetTexture===!1?-1:1):this._equirectMaterial===null&&(this._equirectMaterial=X1());let r=s?this._cubemapMaterial:this._equirectMaterial,a=this._lodMeshes[0];a.material=r;let o=r.uniforms;o.envMap.value=e;let l=this._cubeSize;Jo(n,0,0,3*l,2*l),i.setRenderTarget(n),i.render(a,Pc)}_applyPMREM(e){let n=this._renderer,i=n.autoClear;n.autoClear=!1;let s=this._lodMeshes.length;for(let r=1;r<s;r++)this._applyGGXFilter(e,r-1,r);n.autoClear=i}_applyGGXFilter(e,n,i){let s=this._renderer,r=this._pingPongRenderTarget,a=this._ggxMaterial,o=this._lodMeshes[i];o.material=a;let l=a.uniforms,c=i/(this._lodMeshes.length-1),h=n/(this._lodMeshes.length-1),p=Math.sqrt(c*c-h*h),u=c*1.25,d=p*u,{_lodMax:v}=this,M=this._sizeLods[i],m=3*M*(i>v-jo?i-v+jo:0),f=4*(this._cubeSize-M);l.envMap.value=e.texture,l.roughness.value=d,l.mipInt.value=v-n,Jo(r,m,f,3*M,2*M),s.setRenderTarget(r),s.render(o,Pc),l.envMap.value=r.texture,l.roughness.value=0,l.mipInt.value=v-i,Jo(e,m,f,3*M,2*M),s.setRenderTarget(e),s.render(o,Pc)}_blur(e,n,i,s){let r=this._pingPongRenderTarget,a=Math.min(s,Math.PI)/Math.SQRT2;this._blurPass(e,r,n,i,a),this._blurPass(r,e,i,i,a)}_blurPass(e,n,i,s,r){let a=this._renderer,o=this._blurMaterial,l=this._lodMeshes[s];l.material=o;let c=o.uniforms;c.envMap.value=e.texture,c.sigma.value=r,c.mipInt.value=this._lodMax-i;let h=this._sizeLods[s],p=3*h*(s>this._lodMax-jo?s-this._lodMax+jo:0),u=4*(this._cubeSize-h);Jo(n,p,u,3*h,2*h),a.setRenderTarget(n),a.render(l,Pc)}};function HR(t){let e=[],n=[],i=t,s=t-jo+1+LR;for(let r=0;r<s;r++){let a=Math.pow(2,i);e.push(a);let o=1/(a-2),l=-o,c=1+o,h=[l,l,c,l,c,c,l,l,c,c,l,c],p=6,u=6,d=3,v=new Float32Array(d*u*p),M=new Float32Array(d*u*p);for(let f=0;f<p;f++){let g=f%3*2/3-1,S=f>2?0:-1,_=[g,S,0,g+2/3,S,0,g+2/3,S+1,0,g,S,0,g+2/3,S+1,0,g,S+1,0];v.set(_,d*u*f);for(let E=0;E<u;E++){let T=h[E*2]*2-1,C=h[E*2+1]*2-1;f===0?Ma.set(1,C,T):f===1?Ma.set(-T,1,-C):f===2?Ma.set(-T,C,1):f===3?Ma.set(-1,C,-T):f===4?Ma.set(-T,-1,C):Ma.set(T,C,-1),Ma.toArray(M,(f*u+E)*d)}}let m=new wi;m.setAttribute("position",new Fn(v,d)),m.setAttribute("outputDirection",new Fn(M,d)),n.push(new Rn(m,null)),i>jo&&i--}return{lodMeshes:n,sizeLods:e}}function W1(t,e,n){let i=new Ft(t,e,n);return i.texture.mapping=Tc,i.texture.name="PMREM.cubeUv",i.scissorTest=!0,i}function Jo(t,e,n,i,s){t.viewport.set(e,n,i,s),t.scissor.set(e,n,i,s)}function GR(t,e,n){return new Wt({name:"PMREMGGXConvolution",defines:{GGX_SAMPLES:FR,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/n,CUBEUV_MAX_MIP:`${t}.0`},uniforms:{envMap:{value:null},roughness:{value:0},mipInt:{value:0}},vertexShader:_d(),fragmentShader:`

			precision highp float;
			precision highp int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform float roughness;
			uniform float mipInt;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			#define PI 3.14159265359

			// Van der Corput radical inverse
			float radicalInverse_VdC(uint bits) {
				bits = (bits << 16u) | (bits >> 16u);
				bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
				bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
				bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
				bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
				return float(bits) * 2.3283064365386963e-10; // / 0x100000000
			}

			// Hammersley sequence
			vec2 hammersley(uint i, uint N) {
				return vec2(float(i) / float(N), radicalInverse_VdC(i));
			}

			// GGX VNDF importance sampling (Eric Heitz 2018)
			// "Sampling the GGX Distribution of Visible Normals"
			// https://jcgt.org/published/0007/04/01/
			vec3 importanceSampleGGX_VNDF(vec2 Xi, vec3 V, float roughness) {
				float alpha = roughness * roughness;

				// Section 4.1: Orthonormal basis
				vec3 T1 = vec3(1.0, 0.0, 0.0);
				vec3 T2 = cross(V, T1);

				// Section 4.2: Parameterization of projected area
				float r = sqrt(Xi.x);
				float phi = 2.0 * PI * Xi.y;
				float t1 = r * cos(phi);
				float t2 = r * sin(phi);
				float s = 0.5 * (1.0 + V.z);
				t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

				// Section 4.3: Reprojection onto hemisphere
				vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * V;

				// Section 3.4: Transform back to ellipsoid configuration
				return normalize(vec3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));
			}

			void main() {
				vec3 N = normalize(vOutputDirection);
				vec3 V = N; // Assume view direction equals normal for pre-filtering

				vec3 prefilteredColor = vec3(0.0);
				float totalWeight = 0.0;

				// For very low roughness, just sample the environment directly
				if (roughness < 0.001) {
					gl_FragColor = vec4(bilinearCubeUV(envMap, N, mipInt), 1.0);
					return;
				}

				// Tangent space basis for VNDF sampling
				vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
				vec3 tangent = normalize(cross(up, N));
				vec3 bitangent = cross(N, tangent);

				for(uint i = 0u; i < uint(GGX_SAMPLES); i++) {
					vec2 Xi = hammersley(i, uint(GGX_SAMPLES));

					// For PMREM, V = N, so in tangent space V is always (0, 0, 1)
					vec3 H_tangent = importanceSampleGGX_VNDF(Xi, vec3(0.0, 0.0, 1.0), roughness);

					// Transform H back to world space
					vec3 H = normalize(tangent * H_tangent.x + bitangent * H_tangent.y + N * H_tangent.z);
					vec3 L = normalize(2.0 * dot(V, H) * H - V);

					float NdotL = max(dot(N, L), 0.0);

					if(NdotL > 0.0) {
						// Sample environment at fixed mip level
						// VNDF importance sampling handles the distribution filtering
						vec3 sampleColor = bilinearCubeUV(envMap, L, mipInt);

						// Weight by NdotL for the split-sum approximation
						// VNDF PDF naturally accounts for the visible microfacet distribution
						prefilteredColor += sampleColor * NdotL;
						totalWeight += NdotL;
					}
				}

				if (totalWeight > 0.0) {
					prefilteredColor = prefilteredColor / totalWeight;
				}

				gl_FragColor = vec4(prefilteredColor, 1.0);
			}
		`,blending:Kn,depthTest:!1,depthWrite:!1})}function VR(t,e,n){return new Wt({name:"SphericalGaussianBlur",defines:{SAMPLES:OR,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/n,CUBEUV_MAX_MIP:`${t}.0`},uniforms:{envMap:{value:null},sigma:{value:0},mipInt:{value:0}},vertexShader:_d(),fragmentShader:`

			precision highp float;
			precision highp int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform float sigma;
			uniform float mipInt;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			#define PI 3.14159265359
			#define GOLDEN_ANGLE 2.39996322973

			void main() {

				if ( sigma == 0.0 ) {

					gl_FragColor = vec4( bilinearCubeUV( envMap, vOutputDirection, mipInt ), 1.0 );
					return;

				}

				vec3 outputDirection = normalize( vOutputDirection );

				vec3 up = abs( outputDirection.z ) < 0.999 ? vec3( 0.0, 0.0, 1.0 ) : vec3( 1.0, 0.0, 0.0 );
				vec3 tangent = normalize( cross( up, outputDirection ) );
				vec3 bitangent = cross( outputDirection, tangent );

				// Truncate the kernel at three standard deviations or at the antipode.
				float thetaMax = min( 3.0 * sigma, PI );
				float truncation = 1.0 - exp( - 0.5 * thetaMax * thetaMax / ( sigma * sigma ) );

				vec3 accumColor = vec3( 0.0 );
				float accumWeight = 0.0;

				for ( int i = 0; i < SAMPLES; i ++ ) {

					// Stratified inverse-CDF sampling of the Gaussian, placed on a golden-angle spiral.
					float stratum = ( float( i ) + 0.5 ) / float( SAMPLES );
					float theta = sigma * sqrt( - 2.0 * log( 1.0 - stratum * truncation ) );
					float phi = float( i ) * GOLDEN_ANGLE;

					vec3 offset = cos( phi ) * tangent + sin( phi ) * bitangent;
					vec3 sampleDirection = cos( theta ) * outputDirection + sin( theta ) * offset;

					// Correct the planar sample density to solid angle.
					float weight = sin( theta ) / theta;

					accumColor += weight * bilinearCubeUV( envMap, sampleDirection, mipInt );
					accumWeight += weight;

				}

				gl_FragColor = vec4( accumColor / accumWeight, 1.0 );

			}
		`,blending:Kn,depthTest:!1,depthWrite:!1})}function X1(){return new Wt({name:"EquirectangularToCubeUV",uniforms:{envMap:{value:null}},vertexShader:_d(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;

			#include <common>

			void main() {

				vec3 outputDirection = normalize( vOutputDirection );
				vec2 uv = equirectUv( outputDirection );

				gl_FragColor = vec4( texture2D ( envMap, uv ).rgb, 1.0 );

			}
		`,blending:Kn,depthTest:!1,depthWrite:!1})}function Y1(){return new Wt({name:"CubemapToCubeUV",uniforms:{envMap:{value:null},flipEnvMap:{value:-1}},vertexShader:_d(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			uniform float flipEnvMap;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;

			void main() {

				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );

			}
		`,blending:Kn,depthTest:!1,depthWrite:!1})}function _d(){return`

		precision mediump float;
		precision mediump int;

		attribute vec3 outputDirection;

		varying vec3 vOutputDirection;

		void main() {

			vOutputDirection = outputDirection;
			gl_Position = vec4( position, 1.0 );

		}
	`}var vd=class extends Ft{constructor(e=1,n={}){super(e,e,n),this.isWebGLCubeRenderTarget=!0;let i={width:e,height:e,depth:1},s=[i,i,i,i,i,i];this.texture=new yc(s),this._setTextureOptions(n),this.texture.isRenderTargetTexture=!0}fromEquirectangularTexture(e,n){this.texture.type=n.type,this.texture.colorSpace=n.colorSpace,this.texture.generateMipmaps=n.generateMipmaps,this.texture.minFilter=n.minFilter,this.texture.magFilter=n.magFilter;let i={uniforms:{tEquirect:{value:null}},vertexShader:`

				varying vec3 vWorldDirection;

				vec3 transformDirection( in vec3 dir, in mat4 matrix ) {

					return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );

				}

				void main() {

					vWorldDirection = transformDirection( position, modelMatrix );

					#include <begin_vertex>
					#include <project_vertex>

				}
			`,fragmentShader:`

				uniform sampler2D tEquirect;

				varying vec3 vWorldDirection;

				#include <common>

				void main() {

					vec3 direction = normalize( vWorldDirection );

					vec2 sampleUV = equirectUv( direction );

					gl_FragColor = texture2D( tEquirect, sampleUV );

				}
			`},s=new qo(5,5,5),r=new Wt({name:"CubemapFromEquirect",uniforms:Aa(i.uniforms),vertexShader:i.vertexShader,fragmentShader:i.fragmentShader,side:$t,blending:Kn});r.uniforms.tEquirect.value=n;let a=new Rn(s,r),o=n.minFilter;return n.minFilter===Lr&&(n.minFilter=wt),new Eh(1,10,this).update(e,a),n.minFilter=o,a.geometry.dispose(),a.material.dispose(),this}clear(e,n=!0,i=!0,s=!0){let r=e.getRenderTarget();for(let a=0;a<6;a++)e.setRenderTarget(this,a),e.clear(n,i,s);e.setRenderTarget(r)}};function kR(t){let e=new WeakMap,n=new WeakMap,i=null;function s(u,d=!1){return u==null?null:d?a(u):r(u)}function r(u){if(u&&u.isTexture){let d=u.mapping;if(d===wh||d===Ch)if(e.has(u)){let v=e.get(u).texture;return o(v,u.mapping)}else{let v=u.image;if(v&&v.height>0){let M=new vd(v.height);return M.fromEquirectangularTexture(t,u),e.set(u,M),u.addEventListener("dispose",c),o(M.texture,u.mapping)}else return null}}return u}function a(u){if(u&&u.isTexture){let d=u.mapping,v=d===wh||d===Ch,M=d===Pr||d===Sa;if(v||M){let m=n.get(u),f=m!==void 0?m.texture.pmremVersion:0;if(u.isRenderTargetTexture&&u.pmremVersion!==f)return i===null&&(i=new gd(t)),m=v?i.fromEquirectangular(u,m):i.fromCubemap(u,m),m.texture.pmremVersion=u.pmremVersion,n.set(u,m),m.texture;if(m!==void 0)return m.texture;{let g=u.image;return v&&g&&g.height>0||M&&g&&l(g)?(i===null&&(i=new gd(t)),m=v?i.fromEquirectangular(u):i.fromCubemap(u),m.texture.pmremVersion=u.pmremVersion,n.set(u,m),u.addEventListener("dispose",h),m.texture):null}}}return u}function o(u,d){return d===wh?u.mapping=Pr:d===Ch&&(u.mapping=Sa),u}function l(u){let d=0,v=6;for(let M=0;M<v;M++)u[M]!==void 0&&d++;return d===v}function c(u){let d=u.target;d.removeEventListener("dispose",c);let v=e.get(d);v!==void 0&&(e.delete(d),v.dispose())}function h(u){let d=u.target;d.removeEventListener("dispose",h);let v=n.get(d);v!==void 0&&(n.delete(d),v.dispose())}function p(){e=new WeakMap,n=new WeakMap,i!==null&&(i.dispose(),i=null)}return{get:s,dispose:p}}function WR(t){let e={};function n(i){if(e[i]!==void 0)return e[i];let s=t.getExtension(i);return e[i]=s,s}return{has:function(i){return n(i)!==null},init:function(){n("EXT_color_buffer_float"),n("WEBGL_clip_cull_distance"),n("OES_texture_float_linear"),n("EXT_color_buffer_half_float"),n("WEBGL_multisampled_render_to_texture"),n("WEBGL_render_shared_exponent")},get:function(i){let s=n(i);return s===null&&xa("WebGLRenderer: "+i+" extension not supported."),s}}}function XR(t,e,n,i){let s={},r=new WeakMap;function a(p){let u=p.target;u.index!==null&&e.remove(u.index);for(let v in u.attributes)e.remove(u.attributes[v]);u.removeEventListener("dispose",a),delete s[u.id];let d=r.get(u);d&&(e.remove(d),r.delete(u)),i.releaseStatesOfGeometry(u),u.isInstancedBufferGeometry===!0&&delete u._maxInstanceCount,n.memory.geometries--}function o(p,u){return s[u.id]===!0||(u.addEventListener("dispose",a),s[u.id]=!0,n.memory.geometries++),u}function l(p){let u=p.attributes;for(let d in u)e.update(u[d],t.ARRAY_BUFFER)}function c(p){let u=[],d=p.index,v=p.attributes.position,M=0;if(v===void 0)return;if(d!==null){let g=d.array;M=d.version;for(let S=0,_=g.length;S<_;S+=3){let E=g[S+0],T=g[S+1],C=g[S+2];u.push(E,T,T,C,C,E)}}else{let g=v.array;M=v.version;for(let S=0,_=g.length/3-1;S<_;S+=3){let E=S+0,T=S+1,C=S+2;u.push(E,T,T,C,C,E)}}let m=new(v.count>=65535?gc:mc)(u,1);m.version=M;let f=r.get(p);f&&e.remove(f),r.set(p,m)}function h(p){let u=r.get(p);if(u){let d=p.index;d!==null&&u.version<d.version&&c(p)}else c(p);return r.get(p)}return{get:o,update:l,getWireframeAttribute:h}}function YR(t,e,n){let i;function s(p){i=p}let r,a;function o(p){r=p.type,a=p.bytesPerElement}function l(p,u){t.drawElements(i,u,r,p*a),n.update(u,i,1)}function c(p,u,d){d!==0&&(t.drawElementsInstanced(i,u,r,p*a,d),n.update(u,i,d))}function h(p,u,d){if(d===0)return;e.get("WEBGL_multi_draw").multiDrawElementsWEBGL(i,u,0,r,p,0,d);let M=0;for(let m=0;m<d;m++)M+=u[m];n.update(M,i,1)}this.setMode=s,this.setIndex=o,this.render=l,this.renderInstances=c,this.renderMultiDraw=h}function qR(t){let e={geometries:0,textures:0},n={frame:0,calls:0,triangles:0,points:0,lines:0};function i(r,a,o){switch(n.calls++,a){case t.TRIANGLES:n.triangles+=o*(r/3);break;case t.LINES:n.lines+=o*(r/2);break;case t.LINE_STRIP:n.lines+=o*(r-1);break;case t.LINE_LOOP:n.lines+=o*r;break;case t.POINTS:n.points+=o*r;break;default:Ie("WebGLInfo: Unknown draw mode:",a);break}}function s(){n.calls=0,n.triangles=0,n.points=0,n.lines=0}return{memory:e,render:n,programs:null,autoReset:!0,reset:s,update:i}}function QR(t,e,n){let i=new WeakMap,s=new Ot;function r(a,o,l){let c=a.morphTargetInfluences,h=o.morphAttributes.position||o.morphAttributes.normal||o.morphAttributes.color,p=h!==void 0?h.length:0,u=i.get(o);if(u===void 0||u.count!==p){let b=function(){C.dispose(),i.delete(o),o.removeEventListener("dispose",b)};u!==void 0&&u.texture.dispose();let d=o.morphAttributes.position!==void 0,v=o.morphAttributes.normal!==void 0,M=o.morphAttributes.color!==void 0,m=o.morphAttributes.position||[],f=o.morphAttributes.normal||[],g=o.morphAttributes.color||[],S=0;d===!0&&(S=1),v===!0&&(S=2),M===!0&&(S=3);let _=o.attributes.position.count*S,E=1;_>e.maxTextureSize&&(E=Math.ceil(_/e.maxTextureSize),_=e.maxTextureSize);let T=new Float32Array(_*E*4*p),C=new dc(T,_,E,p);C.type=fi,C.needsUpdate=!0;let y=S*4;for(let R=0;R<p;R++){let N=m[R],F=f[R],k=g[R],B=_*E*4*R;for(let z=0;z<N.count;z++){let Z=z*y;d===!0&&(s.fromBufferAttribute(N,z),T[B+Z+0]=s.x,T[B+Z+1]=s.y,T[B+Z+2]=s.z,T[B+Z+3]=0),v===!0&&(s.fromBufferAttribute(F,z),T[B+Z+4]=s.x,T[B+Z+5]=s.y,T[B+Z+6]=s.z,T[B+Z+7]=0),M===!0&&(s.fromBufferAttribute(k,z),T[B+Z+8]=s.x,T[B+Z+9]=s.y,T[B+Z+10]=s.z,T[B+Z+11]=k.itemSize===4?s.w:1)}}u={count:p,texture:C,size:new Ne(_,E)},i.set(o,u),o.addEventListener("dispose",b)}if(a.isInstancedMesh===!0&&a.morphTexture!==null)l.getUniforms().setValue(t,"morphTexture",a.morphTexture,n);else{let d=0;for(let M=0;M<c.length;M++)d+=c[M];let v=o.morphTargetsRelative?1:1-d;l.getUniforms().setValue(t,"morphTargetBaseInfluence",v),l.getUniforms().setValue(t,"morphTargetInfluences",c)}l.getUniforms().setValue(t,"morphTargetsTexture",u.texture,n),l.getUniforms().setValue(t,"morphTargetsTextureSize",u.size)}return{update:r}}function ZR(t,e,n,i,s){let r=new WeakMap;function a(c){let h=s.render.frame,p=c.geometry,u=e.get(c,p);if(r.get(u)!==h&&(e.update(u),r.set(u,h)),c.isInstancedMesh&&(c.hasEventListener("dispose",l)===!1&&c.addEventListener("dispose",l),r.get(c)!==h&&(n.update(c.instanceMatrix,t.ARRAY_BUFFER),c.instanceColor!==null&&n.update(c.instanceColor,t.ARRAY_BUFFER),r.set(c,h))),c.isSkinnedMesh){let d=c.skeleton;r.get(d)!==h&&(d.update(),r.set(d,h))}return u}function o(){r=new WeakMap}function l(c){let h=c.target;h.removeEventListener("dispose",l),i.releaseStatesOfObject(h),n.remove(h.instanceMatrix),h.instanceColor!==null&&n.remove(h.instanceColor)}return{update:a,dispose:o}}var KR={[f0]:"LINEAR_TONE_MAPPING",[h0]:"REINHARD_TONE_MAPPING",[d0]:"CINEON_TONE_MAPPING",[p0]:"ACES_FILMIC_TONE_MAPPING",[g0]:"AGX_TONE_MAPPING",[v0]:"NEUTRAL_TONE_MAPPING",[m0]:"CUSTOM_TONE_MAPPING"};function JR(t,e,n,i,s,r){let a=new Ft(e,n,{type:t,depthBuffer:s,stencilBuffer:r,samples:i?4:0,storeMultisampledDepthBuffer:!1,storeMultisampledStencilBuffer:!1,resolveDepthBuffer:!1,resolveStencilBuffer:!1}),o=null,l=null,c=new wi;c.setAttribute("position",new Ti([-1,3,0,-1,-1,0,3,-1,0],3)),c.setAttribute("uv",new Ti([0,2,0,0,2,0],2));let h=new fh({uniforms:{tDiffuse:{value:null}},vertexShader:`
			precision highp float;

			uniform mat4 modelViewMatrix;
			uniform mat4 projectionMatrix;

			attribute vec3 position;
			attribute vec2 uv;

			varying vec2 vUv;

			void main() {
				vUv = uv;
				gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
			}`,fragmentShader:`
			precision highp float;

			uniform sampler2D tDiffuse;

			varying vec2 vUv;

			#include <tonemapping_pars_fragment>
			#include <colorspace_pars_fragment>

			void main() {
				gl_FragColor = texture2D( tDiffuse, vUv );

				#ifdef LINEAR_TONE_MAPPING
					gl_FragColor.rgb = LinearToneMapping( gl_FragColor.rgb );
				#elif defined( REINHARD_TONE_MAPPING )
					gl_FragColor.rgb = ReinhardToneMapping( gl_FragColor.rgb );
				#elif defined( CINEON_TONE_MAPPING )
					gl_FragColor.rgb = CineonToneMapping( gl_FragColor.rgb );
				#elif defined( ACES_FILMIC_TONE_MAPPING )
					gl_FragColor.rgb = ACESFilmicToneMapping( gl_FragColor.rgb );
				#elif defined( AGX_TONE_MAPPING )
					gl_FragColor.rgb = AgXToneMapping( gl_FragColor.rgb );
				#elif defined( NEUTRAL_TONE_MAPPING )
					gl_FragColor.rgb = NeutralToneMapping( gl_FragColor.rgb );
				#elif defined( CUSTOM_TONE_MAPPING )
					gl_FragColor.rgb = CustomToneMapping( gl_FragColor.rgb );
				#endif

				#ifdef SRGB_TRANSFER
					gl_FragColor = sRGBTransferOETF( gl_FragColor );
				#endif
			}`,depthTest:!1,depthWrite:!1}),p=new Rn(c,h),u=new Vs(-1,1,1,-1,0,1),d=null,v=null,M=!1,m,f=null,g=[],S=!1;this.setSize=function(_,E){a.setSize(_,E),o!==null&&o.setSize(_,E),l!==null&&l.setSize(_,E);for(let T=0;T<g.length;T++){let C=g[T];C.setSize&&C.setSize(_,E)}},this.setEffects=function(_){g=_,S=g.length>0&&g[0].isRenderPass===!0;let E=a.width,T=a.height;g.length>0&&o===null&&(o=new Ft(E,T,{type:Xi,depthBuffer:!1,stencilBuffer:!1}),l=new Ft(E,T,{type:Xi,depthBuffer:!1,stencilBuffer:!1}));for(let C=0;C<g.length;C++){let y=g[C];y.setSize&&y.setSize(E,T)}},this.begin=function(_,E){if(M||_.toneMapping===ki&&g.length===0)return!1;if(f=E,E!==null){let T=E.width,C=E.height;(a.width!==T||a.height!==C)&&this.setSize(T,C)}return S===!1&&_.setRenderTarget(a),m=_.toneMapping,_.toneMapping=ki,!0},this.hasRenderPass=function(){return S},this.end=function(_,E){_.toneMapping=m,M=!0;let T=a,C=o;for(let y=0;y<g.length;y++){let b=g[y];b.enabled!==!1&&(b.render(_,C,T,E),b.needsSwap!==!1&&(T=C,C=C===o?l:o))}if(d!==_.outputColorSpace||v!==_.toneMapping){d=_.outputColorSpace,v=_.toneMapping,h.defines={},Ke.getTransfer(d)===ct&&(h.defines.SRGB_TRANSFER="");let y=KR[v];y&&(h.defines[y]=""),h.needsUpdate=!0}h.uniforms.tDiffuse.value=T.texture,_.setRenderTarget(f),_.render(p,u),f=null,M=!1},this.isCompositing=function(){return M},this.dispose=function(){a.dispose(),o!==null&&o.dispose(),l!==null&&l.dispose(),c.dispose(),h.dispose()}}var hM=new jt,k0=new Ci(1,1),dM=new dc,pM=new oh,mM=new yc,q1=[],Q1=[],Z1=new Float32Array(16),K1=new Float32Array(9),J1=new Float32Array(4);function el(t,e,n){let i=t[0];if(i<=0||i>0)return t;let s=e*n,r=q1[s];if(r===void 0&&(r=new Float32Array(s),q1[s]=r),e!==0){i.toArray(r,0);for(let a=1,o=0;a!==e;++a)o+=n,t[a].toArray(r,o)}return r}function rn(t,e){if(t.length!==e.length)return!1;for(let n=0,i=t.length;n<i;n++)if(t[n]!==e[n])return!1;return!0}function an(t,e){for(let n=0,i=e.length;n<i;n++)t[n]=e[n]}function Sd(t,e){let n=Q1[e];n===void 0&&(n=new Int32Array(e),Q1[e]=n);for(let i=0;i!==e;++i)n[i]=t.allocateTextureUnit();return n}function jR(t,e){let n=this.cache;n[0]!==e&&(t.uniform1f(this.addr,e),n[0]=e)}function $R(t,e){let n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y)&&(t.uniform2f(this.addr,e.x,e.y),n[0]=e.x,n[1]=e.y);else{if(rn(n,e))return;t.uniform2fv(this.addr,e),an(n,e)}}function eD(t,e){let n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z)&&(t.uniform3f(this.addr,e.x,e.y,e.z),n[0]=e.x,n[1]=e.y,n[2]=e.z);else if(e.r!==void 0)(n[0]!==e.r||n[1]!==e.g||n[2]!==e.b)&&(t.uniform3f(this.addr,e.r,e.g,e.b),n[0]=e.r,n[1]=e.g,n[2]=e.b);else{if(rn(n,e))return;t.uniform3fv(this.addr,e),an(n,e)}}function tD(t,e){let n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z||n[3]!==e.w)&&(t.uniform4f(this.addr,e.x,e.y,e.z,e.w),n[0]=e.x,n[1]=e.y,n[2]=e.z,n[3]=e.w);else{if(rn(n,e))return;t.uniform4fv(this.addr,e),an(n,e)}}function nD(t,e){let n=this.cache,i=e.elements;if(i===void 0){if(rn(n,e))return;t.uniformMatrix2fv(this.addr,!1,e),an(n,e)}else{if(rn(n,i))return;J1.set(i),t.uniformMatrix2fv(this.addr,!1,J1),an(n,i)}}function iD(t,e){let n=this.cache,i=e.elements;if(i===void 0){if(rn(n,e))return;t.uniformMatrix3fv(this.addr,!1,e),an(n,e)}else{if(rn(n,i))return;K1.set(i),t.uniformMatrix3fv(this.addr,!1,K1),an(n,i)}}function sD(t,e){let n=this.cache,i=e.elements;if(i===void 0){if(rn(n,e))return;t.uniformMatrix4fv(this.addr,!1,e),an(n,e)}else{if(rn(n,i))return;Z1.set(i),t.uniformMatrix4fv(this.addr,!1,Z1),an(n,i)}}function rD(t,e){let n=this.cache;n[0]!==e&&(t.uniform1i(this.addr,e),n[0]=e)}function aD(t,e){let n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y)&&(t.uniform2i(this.addr,e.x,e.y),n[0]=e.x,n[1]=e.y);else{if(rn(n,e))return;t.uniform2iv(this.addr,e),an(n,e)}}function oD(t,e){let n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z)&&(t.uniform3i(this.addr,e.x,e.y,e.z),n[0]=e.x,n[1]=e.y,n[2]=e.z);else{if(rn(n,e))return;t.uniform3iv(this.addr,e),an(n,e)}}function lD(t,e){let n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z||n[3]!==e.w)&&(t.uniform4i(this.addr,e.x,e.y,e.z,e.w),n[0]=e.x,n[1]=e.y,n[2]=e.z,n[3]=e.w);else{if(rn(n,e))return;t.uniform4iv(this.addr,e),an(n,e)}}function cD(t,e){let n=this.cache;n[0]!==e&&(t.uniform1ui(this.addr,e),n[0]=e)}function uD(t,e){let n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y)&&(t.uniform2ui(this.addr,e.x,e.y),n[0]=e.x,n[1]=e.y);else{if(rn(n,e))return;t.uniform2uiv(this.addr,e),an(n,e)}}function fD(t,e){let n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z)&&(t.uniform3ui(this.addr,e.x,e.y,e.z),n[0]=e.x,n[1]=e.y,n[2]=e.z);else{if(rn(n,e))return;t.uniform3uiv(this.addr,e),an(n,e)}}function hD(t,e){let n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z||n[3]!==e.w)&&(t.uniform4ui(this.addr,e.x,e.y,e.z,e.w),n[0]=e.x,n[1]=e.y,n[2]=e.z,n[3]=e.w);else{if(rn(n,e))return;t.uniform4uiv(this.addr,e),an(n,e)}}function dD(t,e,n){let i=this.cache,s=n.allocateTextureUnit();i[0]!==s&&(t.uniform1i(this.addr,s),i[0]=s);let r;this.type===t.SAMPLER_2D_SHADOW?(k0.compareFunction=n.isReversedDepthBuffer()?dd:hd,r=k0):r=hM,n.setTexture2D(e||r,s)}function pD(t,e,n){let i=this.cache,s=n.allocateTextureUnit();i[0]!==s&&(t.uniform1i(this.addr,s),i[0]=s),n.setTexture3D(e||pM,s)}function mD(t,e,n){let i=this.cache,s=n.allocateTextureUnit();i[0]!==s&&(t.uniform1i(this.addr,s),i[0]=s),n.setTextureCube(e||mM,s)}function gD(t,e,n){let i=this.cache,s=n.allocateTextureUnit();i[0]!==s&&(t.uniform1i(this.addr,s),i[0]=s),n.setTexture2DArray(e||dM,s)}function vD(t){switch(t){case 5126:return jR;case 35664:return $R;case 35665:return eD;case 35666:return tD;case 35674:return nD;case 35675:return iD;case 35676:return sD;case 5124:case 35670:return rD;case 35667:case 35671:return aD;case 35668:case 35672:return oD;case 35669:case 35673:return lD;case 5125:return cD;case 36294:return uD;case 36295:return fD;case 36296:return hD;case 35678:case 36198:case 36298:case 36306:case 35682:return dD;case 35679:case 36299:case 36307:return pD;case 35680:case 36300:case 36308:case 36293:return mD;case 36289:case 36303:case 36311:case 36292:return gD}}function xD(t,e){t.uniform1fv(this.addr,e)}function yD(t,e){let n=el(e,this.size,2);t.uniform2fv(this.addr,n)}function _D(t,e){let n=el(e,this.size,3);t.uniform3fv(this.addr,n)}function SD(t,e){let n=el(e,this.size,4);t.uniform4fv(this.addr,n)}function AD(t,e){let n=el(e,this.size,4);t.uniformMatrix2fv(this.addr,!1,n)}function MD(t,e){let n=el(e,this.size,9);t.uniformMatrix3fv(this.addr,!1,n)}function ED(t,e){let n=el(e,this.size,16);t.uniformMatrix4fv(this.addr,!1,n)}function TD(t,e){t.uniform1iv(this.addr,e)}function bD(t,e){t.uniform2iv(this.addr,e)}function wD(t,e){t.uniform3iv(this.addr,e)}function CD(t,e){t.uniform4iv(this.addr,e)}function RD(t,e){t.uniform1uiv(this.addr,e)}function DD(t,e){t.uniform2uiv(this.addr,e)}function UD(t,e){t.uniform3uiv(this.addr,e)}function ID(t,e){t.uniform4uiv(this.addr,e)}function BD(t,e,n){let i=this.cache,s=e.length,r=Sd(n,s);rn(i,r)||(t.uniform1iv(this.addr,r),an(i,r));let a;this.type===t.SAMPLER_2D_SHADOW?a=k0:a=hM;for(let o=0;o!==s;++o)n.setTexture2D(e[o]||a,r[o])}function ND(t,e,n){let i=this.cache,s=e.length,r=Sd(n,s);rn(i,r)||(t.uniform1iv(this.addr,r),an(i,r));for(let a=0;a!==s;++a)n.setTexture3D(e[a]||pM,r[a])}function PD(t,e,n){let i=this.cache,s=e.length,r=Sd(n,s);rn(i,r)||(t.uniform1iv(this.addr,r),an(i,r));for(let a=0;a!==s;++a)n.setTextureCube(e[a]||mM,r[a])}function LD(t,e,n){let i=this.cache,s=e.length,r=Sd(n,s);rn(i,r)||(t.uniform1iv(this.addr,r),an(i,r));for(let a=0;a!==s;++a)n.setTexture2DArray(e[a]||dM,r[a])}function OD(t){switch(t){case 5126:return xD;case 35664:return yD;case 35665:return _D;case 35666:return SD;case 35674:return AD;case 35675:return MD;case 35676:return ED;case 5124:case 35670:return TD;case 35667:case 35671:return bD;case 35668:case 35672:return wD;case 35669:case 35673:return CD;case 5125:return RD;case 36294:return DD;case 36295:return UD;case 36296:return ID;case 35678:case 36198:case 36298:case 36306:case 35682:return BD;case 35679:case 36299:case 36307:return ND;case 35680:case 36300:case 36308:case 36293:return PD;case 36289:case 36303:case 36311:case 36292:return LD}}var W0=class{constructor(e,n,i){this.id=e,this.addr=i,this.cache=[],this.type=n.type,this.setValue=vD(n.type)}},X0=class{constructor(e,n,i){this.id=e,this.addr=i,this.cache=[],this.type=n.type,this.size=n.size,this.setValue=OD(n.type)}},Y0=class{constructor(e){this.id=e,this.seq=[],this.map={}}setValue(e,n,i){let s=this.seq;for(let r=0,a=s.length;r!==a;++r){let o=s[r];o.setValue(e,n[o.id],i)}}},G0=/(\w+)(\])?(\[|\.)?/g;function j1(t,e){t.seq.push(e),t.map[e.id]=e}function FD(t,e,n){let i=t.name,s=i.length;for(G0.lastIndex=0;;){let r=G0.exec(i),a=G0.lastIndex,o=r[1],l=r[2]==="]",c=r[3];if(l&&(o=o|0),c===void 0||c==="["&&a+2===s){j1(n,c===void 0?new W0(o,t,e):new X0(o,t,e));break}else{let p=n.map[o];p===void 0&&(p=new Y0(o),j1(n,p)),n=p}}}var $o=class{constructor(e,n){this.seq=[],this.map={};let i=e.getProgramParameter(n,e.ACTIVE_UNIFORMS);for(let a=0;a<i;++a){let o=e.getActiveUniform(n,a),l=e.getUniformLocation(n,o.name);FD(o,l,this)}let s=[],r=[];for(let a of this.seq)a.type===e.SAMPLER_2D_SHADOW||a.type===e.SAMPLER_CUBE_SHADOW||a.type===e.SAMPLER_2D_ARRAY_SHADOW?s.push(a):r.push(a);s.length>0&&(this.seq=s.concat(r))}setValue(e,n,i,s){let r=this.map[n];r!==void 0&&r.setValue(e,i,s)}setOptional(e,n,i){let s=n[i];s!==void 0&&this.setValue(e,i,s)}static upload(e,n,i,s){for(let r=0,a=n.length;r!==a;++r){let o=n[r],l=i[o.id];l.needsUpdate!==!1&&o.setValue(e,l.value,s)}}static seqWithValue(e,n){let i=[];for(let s=0,r=e.length;s!==r;++s){let a=e[s];a.id in n&&i.push(a)}return i}};function $1(t,e,n){let i=t.createShader(e);return t.shaderSource(i,n),t.compileShader(i),i}var zD=37297,HD=0;function GD(t,e){let n=t.split(`
`),i=[],s=Math.max(e-6,0),r=Math.min(e+6,n.length);for(let a=s;a<r;a++){let o=a+1;i.push(`${o===e?">":" "} ${o}: ${n[a]}`)}return i.join(`
`)}var eM=new Pe;function VD(t){Ke._getMatrix(eM,Ke.workingColorSpace,t);let e=`mat3( ${eM.elements.map(n=>n.toFixed(4))} )`;switch(Ke.getTransfer(t)){case uc:return[e,"LinearTransferOETF"];case ct:return[e,"sRGBTransferOETF"];default:return De("WebGLProgram: Unsupported color space: ",t),[e,"LinearTransferOETF"]}}function tM(t,e,n){let i=t.getShaderParameter(e,t.COMPILE_STATUS),r=(t.getShaderInfoLog(e)||"").trim();if(i&&r==="")return"";let a=/ERROR: 0:(\d+)/.exec(r);if(a){let o=parseInt(a[1]);return n.toUpperCase()+`

`+r+`

`+GD(t.getShaderSource(e),o)}else return r}function kD(t,e){let n=VD(e);return[`vec4 ${t}( vec4 value ) {`,`	return ${n[1]}( vec4( value.rgb * ${n[0]}, value.a ) );`,"}"].join(`
`)}var WD={[f0]:"Linear",[h0]:"Reinhard",[d0]:"Cineon",[p0]:"ACESFilmic",[g0]:"AgX",[v0]:"Neutral",[m0]:"Custom"};function XD(t,e){let n=WD[e];return n===void 0?(De("WebGLProgram: Unsupported toneMapping:",e),"vec3 "+t+"( vec3 color ) { return LinearToneMapping( color ); }"):"vec3 "+t+"( vec3 color ) { return "+n+"ToneMapping( color ); }"}var md=new G;function YD(){Ke.getLuminanceCoefficients(md);let t=md.x.toFixed(4),e=md.y.toFixed(4),n=md.z.toFixed(4);return["float luminance( const in vec3 rgb ) {",`	const vec3 weights = vec3( ${t}, ${e}, ${n} );`,"	return dot( weights, rgb );","}"].join(`
`)}function qD(t){return[t.extensionClipCullDistance?"#extension GL_ANGLE_clip_cull_distance : require":"",t.extensionMultiDraw?"#extension GL_ANGLE_multi_draw : require":""].filter(Oc).join(`
`)}function QD(t){let e=[];for(let n in t){let i=t[n];i!==!1&&e.push("#define "+n+" "+i)}return e.join(`
`)}function ZD(t,e){let n={},i=t.getProgramParameter(e,t.ACTIVE_ATTRIBUTES);for(let s=0;s<i;s++){let r=t.getActiveAttrib(e,s),a=r.name,o=1;r.type===t.FLOAT_MAT2&&(o=2),r.type===t.FLOAT_MAT3&&(o=3),r.type===t.FLOAT_MAT4&&(o=4),n[a]={type:r.type,location:t.getAttribLocation(e,a),locationSize:o}}return n}function Oc(t){return t!==""}function nM(t,e){let n=e.numSpotLightShadows+e.numSpotLightMaps-e.numSpotLightShadowsWithMaps;return t.replace(/NUM_SUN_LIGHTS/g,e.numSunLights).replace(/NUM_DIR_LIGHTS/g,e.numDirLights).replace(/NUM_SPOT_LIGHTS/g,e.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g,e.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g,n).replace(/NUM_RECT_AREA_LIGHTS/g,e.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g,e.numPointLights).replace(/NUM_HEMI_LIGHTS/g,e.numHemiLights).replace(/NUM_SUN_LIGHT_SHADOWS/g,e.numSunLightShadows).replace(/NUM_DIR_LIGHT_SHADOWS/g,e.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g,e.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g,e.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g,e.numPointLightShadows)}function iM(t,e){return t.replace(/NUM_CLIPPING_PLANES/g,e.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g,e.numClippingPlanes-e.numClipIntersection)}var KD=/^[ \t]*#include +<([\w\d./]+)>/gm;function q0(t){return t.replace(KD,jD)}var JD=new Map;function jD(t,e){let n=He[e];if(n===void 0){let i=JD.get(e);if(i!==void 0)n=He[i],De('WebGLRenderer: Shader chunk "%s" has been deprecated. Use "%s" instead.',e,i);else throw new Error("THREE.WebGLProgram: Can not resolve #include <"+e+">")}return q0(n)}var $D=/#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;function sM(t){return t.replace($D,eU)}function eU(t,e,n,i){let s="";for(let r=parseInt(e);r<parseInt(n);r++)s+=i.replace(/\[\s*i\s*\]/g,"[ "+r+" ]").replace(/UNROLLED_LOOP_INDEX/g,r);return s}function rM(t){let e=`precision ${t.precision} float;
	precision ${t.precision} int;
	precision ${t.precision} sampler2D;
	precision ${t.precision} samplerCube;
	precision ${t.precision} sampler3D;
	precision ${t.precision} sampler2DArray;
	precision ${t.precision} sampler2DShadow;
	precision ${t.precision} samplerCubeShadow;
	precision ${t.precision} sampler2DArrayShadow;
	precision ${t.precision} isampler2D;
	precision ${t.precision} isampler3D;
	precision ${t.precision} isamplerCube;
	precision ${t.precision} isampler2DArray;
	precision ${t.precision} usampler2D;
	precision ${t.precision} usampler3D;
	precision ${t.precision} usamplerCube;
	precision ${t.precision} usampler2DArray;
	`;return t.precision==="highp"?e+=`
#define HIGH_PRECISION`:t.precision==="mediump"?e+=`
#define MEDIUM_PRECISION`:t.precision==="lowp"&&(e+=`
#define LOW_PRECISION`),e}var tU={[Ec]:"SHADOWMAP_TYPE_PCF",[Qo]:"SHADOWMAP_TYPE_VSM"};function nU(t){return tU[t.shadowMapType]||"SHADOWMAP_TYPE_BASIC"}var iU={[Pr]:"ENVMAP_TYPE_CUBE",[Sa]:"ENVMAP_TYPE_CUBE",[Tc]:"ENVMAP_TYPE_CUBE_UV"};function sU(t){return t.envMap===!1?"ENVMAP_TYPE_CUBE":iU[t.envMapMode]||"ENVMAP_TYPE_CUBE"}var rU={[Sa]:"ENVMAP_MODE_REFRACTION"};function aU(t){return t.envMap===!1?"ENVMAP_MODE_REFLECTION":rU[t.envMapMode]||"ENVMAP_MODE_REFLECTION"}var oU={[u0]:"ENVMAP_BLENDING_MULTIPLY",[M1]:"ENVMAP_BLENDING_MIX",[E1]:"ENVMAP_BLENDING_ADD"};function lU(t){return t.envMap===!1?"ENVMAP_BLENDING_NONE":oU[t.combine]||"ENVMAP_BLENDING_NONE"}function cU(t){let e=t.envMapCubeUVHeight;if(e===null)return null;let n=Math.log2(e)-2,i=1/e;return{texelWidth:1/(3*Math.max(Math.pow(2,n),112)),texelHeight:i,maxMip:n}}function uU(t,e,n,i){let s=t.getContext(),r=n.defines,a=n.vertexShader,o=n.fragmentShader,l=nU(n),c=sU(n),h=aU(n),p=lU(n),u=cU(n),d=qD(n),v=QD(r),M=s.createProgram(),m,f,g=n.glslVersion?"#version "+n.glslVersion+`
`:"";n.isRawShaderMaterial?(m=["#define SHADER_TYPE "+n.shaderType,"#define SHADER_NAME "+n.shaderName,v].filter(Oc).join(`
`),m.length>0&&(m+=`
`),f=["#define SHADER_TYPE "+n.shaderType,"#define SHADER_NAME "+n.shaderName,v].filter(Oc).join(`
`),f.length>0&&(f+=`
`)):(m=[rM(n),"#define SHADER_TYPE "+n.shaderType,"#define SHADER_NAME "+n.shaderName,v,n.extensionClipCullDistance?"#define USE_CLIP_DISTANCE":"",n.batching?"#define USE_BATCHING":"",n.batchingColor?"#define USE_BATCHING_COLOR":"",n.instancing?"#define USE_INSTANCING":"",n.instancingColor?"#define USE_INSTANCING_COLOR":"",n.instancingMorph?"#define USE_INSTANCING_MORPH":"",n.useFog&&n.fog?"#define USE_FOG":"",n.useFog&&n.fogExp2?"#define FOG_EXP2":"",n.map?"#define USE_MAP":"",n.envMap?"#define USE_ENVMAP":"",n.envMap?"#define "+h:"",n.lightMap?"#define USE_LIGHTMAP":"",n.aoMap?"#define USE_AOMAP":"",n.bumpMap?"#define USE_BUMPMAP":"",n.normalMap?"#define USE_NORMALMAP":"",n.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",n.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",n.displacementMap?"#define USE_DISPLACEMENTMAP":"",n.emissiveMap?"#define USE_EMISSIVEMAP":"",n.anisotropy?"#define USE_ANISOTROPY":"",n.anisotropyMap?"#define USE_ANISOTROPYMAP":"",n.clearcoatMap?"#define USE_CLEARCOATMAP":"",n.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",n.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",n.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",n.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",n.specularMap?"#define USE_SPECULARMAP":"",n.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",n.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",n.roughnessMap?"#define USE_ROUGHNESSMAP":"",n.metalnessMap?"#define USE_METALNESSMAP":"",n.alphaMap?"#define USE_ALPHAMAP":"",n.alphaHash?"#define USE_ALPHAHASH":"",n.transmission?"#define USE_TRANSMISSION":"",n.transmissionMap?"#define USE_TRANSMISSIONMAP":"",n.thicknessMap?"#define USE_THICKNESSMAP":"",n.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",n.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",n.mapUv?"#define MAP_UV "+n.mapUv:"",n.alphaMapUv?"#define ALPHAMAP_UV "+n.alphaMapUv:"",n.lightMapUv?"#define LIGHTMAP_UV "+n.lightMapUv:"",n.aoMapUv?"#define AOMAP_UV "+n.aoMapUv:"",n.emissiveMapUv?"#define EMISSIVEMAP_UV "+n.emissiveMapUv:"",n.bumpMapUv?"#define BUMPMAP_UV "+n.bumpMapUv:"",n.normalMapUv?"#define NORMALMAP_UV "+n.normalMapUv:"",n.displacementMapUv?"#define DISPLACEMENTMAP_UV "+n.displacementMapUv:"",n.metalnessMapUv?"#define METALNESSMAP_UV "+n.metalnessMapUv:"",n.roughnessMapUv?"#define ROUGHNESSMAP_UV "+n.roughnessMapUv:"",n.anisotropyMapUv?"#define ANISOTROPYMAP_UV "+n.anisotropyMapUv:"",n.clearcoatMapUv?"#define CLEARCOATMAP_UV "+n.clearcoatMapUv:"",n.clearcoatNormalMapUv?"#define CLEARCOAT_NORMALMAP_UV "+n.clearcoatNormalMapUv:"",n.clearcoatRoughnessMapUv?"#define CLEARCOAT_ROUGHNESSMAP_UV "+n.clearcoatRoughnessMapUv:"",n.iridescenceMapUv?"#define IRIDESCENCEMAP_UV "+n.iridescenceMapUv:"",n.iridescenceThicknessMapUv?"#define IRIDESCENCE_THICKNESSMAP_UV "+n.iridescenceThicknessMapUv:"",n.sheenColorMapUv?"#define SHEEN_COLORMAP_UV "+n.sheenColorMapUv:"",n.sheenRoughnessMapUv?"#define SHEEN_ROUGHNESSMAP_UV "+n.sheenRoughnessMapUv:"",n.specularMapUv?"#define SPECULARMAP_UV "+n.specularMapUv:"",n.specularColorMapUv?"#define SPECULAR_COLORMAP_UV "+n.specularColorMapUv:"",n.specularIntensityMapUv?"#define SPECULAR_INTENSITYMAP_UV "+n.specularIntensityMapUv:"",n.transmissionMapUv?"#define TRANSMISSIONMAP_UV "+n.transmissionMapUv:"",n.thicknessMapUv?"#define THICKNESSMAP_UV "+n.thicknessMapUv:"",n.vertexTangents&&n.flatShading===!1?"#define USE_TANGENT":"",n.vertexNormals?"#define HAS_NORMAL":"",n.vertexColors?"#define USE_COLOR":"",n.vertexAlphas?"#define USE_COLOR_ALPHA":"",n.vertexUv1s?"#define USE_UV1":"",n.vertexUv2s?"#define USE_UV2":"",n.vertexUv3s?"#define USE_UV3":"",n.pointsUvs?"#define USE_POINTS_UV":"",n.flatShading?"#define FLAT_SHADED":"",n.skinning?"#define USE_SKINNING":"",n.morphTargets?"#define USE_MORPHTARGETS":"",n.morphNormals&&n.flatShading===!1?"#define USE_MORPHNORMALS":"",n.morphColors?"#define USE_MORPHCOLORS":"",n.morphTargetsCount>0?"#define MORPHTARGETS_TEXTURE_STRIDE "+n.morphTextureStride:"",n.morphTargetsCount>0?"#define MORPHTARGETS_COUNT "+n.morphTargetsCount:"",n.doubleSided?"#define DOUBLE_SIDED":"",n.flipSided?"#define FLIP_SIDED":"",n.shadowMapEnabled?"#define USE_SHADOWMAP":"",n.shadowMapEnabled?"#define "+l:"",n.sizeAttenuation?"#define USE_SIZEATTENUATION":"",n.numLightProbes>0?"#define USE_LIGHT_PROBES":"",n.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",n.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 modelMatrix;","uniform mat4 modelViewMatrix;","uniform mat4 projectionMatrix;","uniform mat4 viewMatrix;","uniform mat3 normalMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;","#ifdef USE_INSTANCING","	attribute mat4 instanceMatrix;","#endif","#ifdef USE_INSTANCING_COLOR","	attribute vec3 instanceColor;","#endif","#ifdef USE_INSTANCING_MORPH","	uniform sampler2D morphTexture;","#endif","attribute vec3 position;","attribute vec3 normal;","attribute vec2 uv;","#ifdef USE_UV1","	attribute vec2 uv1;","#endif","#ifdef USE_UV2","	attribute vec2 uv2;","#endif","#ifdef USE_UV3","	attribute vec2 uv3;","#endif","#ifdef USE_TANGENT","	attribute vec4 tangent;","#endif","#if defined( USE_COLOR_ALPHA )","	attribute vec4 color;","#elif defined( USE_COLOR )","	attribute vec3 color;","#endif","#ifdef USE_SKINNING","	attribute vec4 skinIndex;","	attribute vec4 skinWeight;","#endif",`
`].filter(Oc).join(`
`),f=[rM(n),"#define SHADER_TYPE "+n.shaderType,"#define SHADER_NAME "+n.shaderName,v,n.useFog&&n.fog?"#define USE_FOG":"",n.useFog&&n.fogExp2?"#define FOG_EXP2":"",n.alphaToCoverage?"#define ALPHA_TO_COVERAGE":"",n.map?"#define USE_MAP":"",n.matcap?"#define USE_MATCAP":"",n.envMap?"#define USE_ENVMAP":"",n.envMap?"#define "+c:"",n.envMap?"#define "+h:"",n.envMap?"#define "+p:"",u?"#define CUBEUV_TEXEL_WIDTH "+u.texelWidth:"",u?"#define CUBEUV_TEXEL_HEIGHT "+u.texelHeight:"",u?"#define CUBEUV_MAX_MIP "+u.maxMip+".0":"",n.lightMap?"#define USE_LIGHTMAP":"",n.aoMap?"#define USE_AOMAP":"",n.bumpMap?"#define USE_BUMPMAP":"",n.normalMap?"#define USE_NORMALMAP":"",n.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",n.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",n.packedNormalMap?"#define USE_PACKED_NORMALMAP":"",n.emissiveMap?"#define USE_EMISSIVEMAP":"",n.anisotropy?"#define USE_ANISOTROPY":"",n.anisotropyMap?"#define USE_ANISOTROPYMAP":"",n.clearcoat?"#define USE_CLEARCOAT":"",n.clearcoatMap?"#define USE_CLEARCOATMAP":"",n.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",n.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",n.dispersion?"#define USE_DISPERSION":"",n.retroreflection?"#define USE_RETROREFLECTION":"",n.iridescence?"#define USE_IRIDESCENCE":"",n.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",n.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",n.specularMap?"#define USE_SPECULARMAP":"",n.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",n.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",n.roughnessMap?"#define USE_ROUGHNESSMAP":"",n.metalnessMap?"#define USE_METALNESSMAP":"",n.alphaMap?"#define USE_ALPHAMAP":"",n.alphaTest?"#define USE_ALPHATEST":"",n.alphaHash?"#define USE_ALPHAHASH":"",n.sheen?"#define USE_SHEEN":"",n.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",n.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",n.transmission?"#define USE_TRANSMISSION":"",n.transmissionMap?"#define USE_TRANSMISSIONMAP":"",n.thicknessMap?"#define USE_THICKNESSMAP":"",n.vertexTangents&&n.flatShading===!1?"#define USE_TANGENT":"",n.vertexColors||n.instancingColor?"#define USE_COLOR":"",n.vertexAlphas||n.batchingColor?"#define USE_COLOR_ALPHA":"",n.vertexUv1s?"#define USE_UV1":"",n.vertexUv2s?"#define USE_UV2":"",n.vertexUv3s?"#define USE_UV3":"",n.pointsUvs?"#define USE_POINTS_UV":"",n.gradientMap?"#define USE_GRADIENTMAP":"",n.flatShading?"#define FLAT_SHADED":"",n.doubleSided?"#define DOUBLE_SIDED":"",n.flipSided?"#define FLIP_SIDED":"",n.shadowMapEnabled?"#define USE_SHADOWMAP":"",n.shadowMapEnabled?"#define "+l:"",n.premultipliedAlpha?"#define PREMULTIPLIED_ALPHA":"",n.numLightProbes>0?"#define USE_LIGHT_PROBES":"",n.numLightProbeGrids>0?"#define USE_LIGHT_PROBES_GRID":"",n.decodeVideoTexture?"#define DECODE_VIDEO_TEXTURE":"",n.decodeVideoTextureEmissive?"#define DECODE_VIDEO_TEXTURE_EMISSIVE":"",n.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",n.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 viewMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;",n.toneMapping!==ki?"#define TONE_MAPPING":"",n.toneMapping!==ki?He.tonemapping_pars_fragment:"",n.toneMapping!==ki?XD("toneMapping",n.toneMapping):"",n.dithering?"#define DITHERING":"",n.opaque?"#define OPAQUE":"",He.colorspace_pars_fragment,kD("linearToOutputTexel",n.outputColorSpace),YD(),n.useDepthPacking?"#define DEPTH_PACKING "+n.depthPacking:"",`
`].filter(Oc).join(`
`)),a=q0(a),a=nM(a,n),a=iM(a,n),o=q0(o),o=nM(o,n),o=iM(o,n),a=sM(a),o=sM(o),n.isRawShaderMaterial!==!0&&(g=`#version 300 es
`,m=[d,"#define attribute in","#define varying out","#define texture2D texture"].join(`
`)+`
`+m,f=["#define varying in",n.glslVersion===Bc?"":"layout(location = 0) out highp vec4 pc_fragColor;",n.glslVersion===Bc?"":"#define gl_FragColor pc_fragColor","#define gl_FragDepthEXT gl_FragDepth","#define texture2D texture","#define textureCube texture","#define texture2DProj textureProj","#define texture2DLodEXT textureLod","#define texture2DProjLodEXT textureProjLod","#define textureCubeLodEXT textureLod","#define texture2DGradEXT textureGrad","#define texture2DProjGradEXT textureProjGrad","#define textureCubeGradEXT textureGrad"].join(`
`)+`
`+f);let S=g+m+a,_=g+f+o,E=$1(s,s.VERTEX_SHADER,S),T=$1(s,s.FRAGMENT_SHADER,_);s.attachShader(M,E),s.attachShader(M,T),n.index0AttributeName!==void 0?s.bindAttribLocation(M,0,n.index0AttributeName):n.hasPositionAttribute===!0&&s.bindAttribLocation(M,0,"position"),s.linkProgram(M);function C(N){if(t.debug.checkShaderErrors){let F=s.getProgramInfoLog(M)||"",k=s.getShaderInfoLog(E)||"",B=s.getShaderInfoLog(T)||"",z=F.trim(),Z=k.trim(),q=B.trim(),ie=!0,W=!0;if(s.getProgramParameter(M,s.LINK_STATUS)===!1)if(ie=!1,typeof t.debug.onShaderError=="function")t.debug.onShaderError(s,M,E,T);else{let $=tM(s,E,"vertex"),te=tM(s,T,"fragment");Ie("WebGLProgram: Shader Error "+s.getError()+" - VALIDATE_STATUS "+s.getProgramParameter(M,s.VALIDATE_STATUS)+`

Material Name: `+N.name+`
Material Type: `+N.type+`

Program Info Log: `+z+`
`+$+`
`+te)}else z!==""?De("WebGLProgram: Program Info Log:",z):(Z===""||q==="")&&(W=!1);W&&(N.diagnostics={runnable:ie,programLog:z,vertexShader:{log:Z,prefix:m},fragmentShader:{log:q,prefix:f}})}s.deleteShader(E),s.deleteShader(T),y=new $o(s,M),b=ZD(s,M)}let y;this.getUniforms=function(){return y===void 0&&C(this),y};let b;this.getAttributes=function(){return b===void 0&&C(this),b};let R=n.rendererExtensionParallelShaderCompile===!1;return this.isReady=function(){return R===!1&&(R=s.getProgramParameter(M,zD)),R},this.destroy=function(){i.releaseStatesOfProgram(this),s.deleteProgram(M),this.program=void 0},this.type=n.shaderType,this.name=n.shaderName,this.id=HD++,this.cacheKey=e,this.usedTimes=1,this.program=M,this.vertexShader=E,this.fragmentShader=T,this}var fU=0,Q0=class{constructor(){this.shaderCache=new Map,this.materialCache=new Map}update(e,n,i){let s=this._getShaderCacheForMaterial(e);return s.has(n)===!1&&(s.add(n),n.usedTimes++),s.has(i)===!1&&(s.add(i),i.usedTimes++),this}remove(e){let n=this.materialCache.get(e);for(let i of n)i.usedTimes--,i.usedTimes===0&&this.shaderCache.delete(i.code);return this.materialCache.delete(e),this}getVertexShaderStage(e){return this._getShaderStage(e.vertexShader)}getFragmentShaderStage(e){return this._getShaderStage(e.fragmentShader)}dispose(){this.shaderCache.clear(),this.materialCache.clear()}_getShaderCacheForMaterial(e){let n=this.materialCache,i=n.get(e);return i===void 0&&(i=new Set,n.set(e,i)),i}_getShaderStage(e){let n=this.shaderCache,i=n.get(e);return i===void 0&&(i=new Z0(e),n.set(e,i)),i}},Z0=class{constructor(e){this.id=fU++,this.code=e,this.usedTimes=0}};function hU(t){return t===Fr||t===Uc||t===Ic}function dU(t,e,n,i,s,r){let a=new pc,o=new Q0,l=new Set,c=[],h=new Map,p=i.logarithmicDepthBuffer,u=i.precision,d={MeshDepthMaterial:"depth",MeshDistanceMaterial:"distance",MeshNormalMaterial:"normal",MeshBasicMaterial:"basic",MeshLambertMaterial:"lambert",MeshPhongMaterial:"phong",MeshToonMaterial:"toon",MeshStandardMaterial:"physical",MeshPhysicalMaterial:"physical",MeshMatcapMaterial:"matcap",LineBasicMaterial:"basic",LineDashedMaterial:"dashed",PointsMaterial:"points",ShadowMaterial:"shadow",SpriteMaterial:"sprite"};function v(y){return l.add(y),y===0?"uv":`uv${y}`}function M(y,b,R,N,F,k){let B=N.fog,z=F.geometry,Z=y.isMeshStandardMaterial||y.isMeshLambertMaterial||y.isMeshPhongMaterial?N.environment:null,q=y.isMeshStandardMaterial||y.isMeshLambertMaterial&&!y.envMap||y.isMeshPhongMaterial&&!y.envMap,ie=e.get(y.envMap||Z,q),W=ie&&ie.mapping===Tc?ie.image.height:null,$=d[y.type];y.precision!==null&&(u=i.getMaxPrecision(y.precision),u!==y.precision&&De("WebGLProgram.getParameters:",y.precision,"not supported, using",u,"instead."));let te=z.morphAttributes.position||z.morphAttributes.normal||z.morphAttributes.color,we=te!==void 0?te.length:0,Me=0;z.morphAttributes.position!==void 0&&(Me=1),z.morphAttributes.normal!==void 0&&(Me=2),z.morphAttributes.color!==void 0&&(Me=3);let ut,qe,$e,X;if($){let St=xs[$];ut=St.vertexShader,qe=St.fragmentShader}else{ut=y.vertexShader,qe=y.fragmentShader;let St=o.getVertexShaderStage(y),at=o.getFragmentShaderStage(y);o.update(y,St,at),$e=St.id,X=at.id}let ee=t.getRenderTarget(),ue=t.state.buffers.depth.getReversed(),Se=F.isInstancedMesh===!0,ge=F.isBatchedMesh===!0,Fe=!!y.map,en=!!y.matcap,Ve=!!ie,tt=!!y.aoMap,_t=!!y.lightMap,Qe=!!y.bumpMap&&y.wireframe===!1,Ut=!!y.normalMap,on=!!y.displacementMap,Hn=!!y.emissiveMap,Pt=!!y.metalnessMap,Yt=!!y.roughnessMap,I=y.anisotropy>0,Sn=y.clearcoat>0,ft=y.dispersion>0,w=y.retroreflectivity>0,x=y.iridescence>0,P=y.sheen>0,H=y.transmission>0,Y=I&&!!y.anisotropyMap,se=Sn&&!!y.clearcoatMap,re=Sn&&!!y.clearcoatNormalMap,Q=Sn&&!!y.clearcoatRoughnessMap,j=x&&!!y.iridescenceMap,ae=x&&!!y.iridescenceThicknessMap,Te=P&&!!y.sheenColorMap,fe=P&&!!y.sheenRoughnessMap,oe=!!y.specularMap,be=!!y.specularColorMap,Re=!!y.specularIntensityMap,Le=H&&!!y.transmissionMap,U=H&&!!y.thicknessMap,le=!!y.gradientMap,K=!!y.alphaMap,ce=y.alphaTest>0,me=!!y.alphaHash,ne=!!y.extensions,Ce=ki;y.toneMapped&&(ee===null||ee.isXRRenderTarget===!0)&&(Ce=t.toneMapping);let Ae={shaderID:$,shaderType:y.type,shaderName:y.name,vertexShader:ut,fragmentShader:qe,defines:y.defines,customVertexShaderID:$e,customFragmentShaderID:X,isRawShaderMaterial:y.isRawShaderMaterial===!0,glslVersion:y.glslVersion,precision:u,batching:ge,batchingColor:ge&&F._colorsTexture!==null,instancing:Se,instancingColor:Se&&F.instanceColor!==null,instancingMorph:Se&&F.morphTexture!==null,outputColorSpace:ee===null?t.outputColorSpace:ee.isXRRenderTarget===!0?ee.texture.colorSpace:Ke.workingColorSpace,alphaToCoverage:!!y.alphaToCoverage,map:Fe,matcap:en,envMap:Ve,envMapMode:Ve&&ie.mapping,envMapCubeUVHeight:W,aoMap:tt,lightMap:_t,bumpMap:Qe,normalMap:Ut,displacementMap:on,emissiveMap:Hn,normalMapObjectSpace:Ut&&y.normalMapType===b1,normalMapTangentSpace:Ut&&y.normalMapType===b0,packedNormalMap:Ut&&y.normalMapType===b0&&hU(y.normalMap.format),metalnessMap:Pt,roughnessMap:Yt,anisotropy:I,anisotropyMap:Y,clearcoat:Sn,clearcoatMap:se,clearcoatNormalMap:re,clearcoatRoughnessMap:Q,dispersion:ft,retroreflection:w,iridescence:x,iridescenceMap:j,iridescenceThicknessMap:ae,sheen:P,sheenColorMap:Te,sheenRoughnessMap:fe,specularMap:oe,specularColorMap:be,specularIntensityMap:Re,transmission:H,transmissionMap:Le,thicknessMap:U,gradientMap:le,opaque:y.transparent===!1&&y.blending===Zo&&y.alphaToCoverage===!1,alphaMap:K,alphaTest:ce,alphaHash:me,combine:y.combine,mapUv:Fe&&v(y.map.channel),aoMapUv:tt&&v(y.aoMap.channel),lightMapUv:_t&&v(y.lightMap.channel),bumpMapUv:Qe&&v(y.bumpMap.channel),normalMapUv:Ut&&v(y.normalMap.channel),displacementMapUv:on&&v(y.displacementMap.channel),emissiveMapUv:Hn&&v(y.emissiveMap.channel),metalnessMapUv:Pt&&v(y.metalnessMap.channel),roughnessMapUv:Yt&&v(y.roughnessMap.channel),anisotropyMapUv:Y&&v(y.anisotropyMap.channel),clearcoatMapUv:se&&v(y.clearcoatMap.channel),clearcoatNormalMapUv:re&&v(y.clearcoatNormalMap.channel),clearcoatRoughnessMapUv:Q&&v(y.clearcoatRoughnessMap.channel),iridescenceMapUv:j&&v(y.iridescenceMap.channel),iridescenceThicknessMapUv:ae&&v(y.iridescenceThicknessMap.channel),sheenColorMapUv:Te&&v(y.sheenColorMap.channel),sheenRoughnessMapUv:fe&&v(y.sheenRoughnessMap.channel),specularMapUv:oe&&v(y.specularMap.channel),specularColorMapUv:be&&v(y.specularColorMap.channel),specularIntensityMapUv:Re&&v(y.specularIntensityMap.channel),transmissionMapUv:Le&&v(y.transmissionMap.channel),thicknessMapUv:U&&v(y.thicknessMap.channel),alphaMapUv:K&&v(y.alphaMap.channel),vertexTangents:!!z.attributes.tangent&&(Ut||I),vertexNormals:!!z.attributes.normal,vertexColors:y.vertexColors,vertexAlphas:y.vertexColors===!0&&!!z.attributes.color&&z.attributes.color.itemSize===4,pointsUvs:F.isPoints===!0&&!!z.attributes.uv&&(Fe||K),fog:!!B,useFog:y.fog===!0,fogExp2:!!B&&B.isFogExp2,flatShading:y.wireframe===!1&&(y.flatShading===!0||z.attributes.normal===void 0&&Ut===!1&&(y.isMeshLambertMaterial||y.isMeshPhongMaterial||y.isMeshStandardMaterial||y.isMeshPhysicalMaterial)),sizeAttenuation:y.sizeAttenuation===!0,logarithmicDepthBuffer:p,reversedDepthBuffer:ue,skinning:F.isSkinnedMesh===!0,hasPositionAttribute:z.attributes.position!==void 0,morphTargets:z.morphAttributes.position!==void 0,morphNormals:z.morphAttributes.normal!==void 0,morphColors:z.morphAttributes.color!==void 0,morphTargetsCount:we,morphTextureStride:Me,numSunLights:b.sun.length,numDirLights:b.directional.length,numPointLights:b.point.length,numSpotLights:b.spot.length,numSpotLightMaps:b.spotLightMap.length,numRectAreaLights:b.rectArea.length,numHemiLights:b.hemi.length,numSunLightShadows:b.sunShadowMap.length,numDirLightShadows:b.directionalShadowMap.length,numPointLightShadows:b.pointShadowMap.length,numSpotLightShadows:b.spotShadowMap.length,numSpotLightShadowsWithMaps:b.numSpotLightShadowsWithMaps,numLightProbes:b.numLightProbes,numLightProbeGrids:k.length,numClippingPlanes:r.numPlanes,numClipIntersection:r.numIntersection,dithering:y.dithering,shadowMapEnabled:t.shadowMap.enabled&&R.length>0,shadowMapType:t.shadowMap.type,toneMapping:Ce,decodeVideoTexture:Fe&&y.map.isVideoTexture===!0&&Ke.getTransfer(y.map.colorSpace)===ct,decodeVideoTextureEmissive:Hn&&y.emissiveMap.isVideoTexture===!0&&Ke.getTransfer(y.emissiveMap.colorSpace)===ct,premultipliedAlpha:y.premultipliedAlpha,doubleSided:y.side===zn,flipSided:y.side===$t,useDepthPacking:y.depthPacking>=0,depthPacking:y.depthPacking||0,index0AttributeName:y.index0AttributeName,extensionClipCullDistance:ne&&y.extensions.clipCullDistance===!0&&n.has("WEBGL_clip_cull_distance"),extensionMultiDraw:(ne&&y.extensions.multiDraw===!0||ge)&&n.has("WEBGL_multi_draw"),rendererExtensionParallelShaderCompile:n.has("KHR_parallel_shader_compile"),customProgramCacheKey:y.customProgramCacheKey()};return Ae.vertexUv1s=l.has(1),Ae.vertexUv2s=l.has(2),Ae.vertexUv3s=l.has(3),l.clear(),Ae}function m(y){let b=[];if(y.shaderID?b.push(y.shaderID):(b.push(y.customVertexShaderID),b.push(y.customFragmentShaderID)),y.defines!==void 0)for(let R in y.defines)b.push(R),b.push(y.defines[R]);return y.isRawShaderMaterial===!1&&(f(b,y),g(b,y),b.push(t.outputColorSpace)),b.push(y.customProgramCacheKey),b.join()}function f(y,b){y.push(b.precision),y.push(b.outputColorSpace),y.push(b.envMapMode),y.push(b.envMapCubeUVHeight),y.push(b.mapUv),y.push(b.alphaMapUv),y.push(b.lightMapUv),y.push(b.aoMapUv),y.push(b.bumpMapUv),y.push(b.normalMapUv),y.push(b.displacementMapUv),y.push(b.emissiveMapUv),y.push(b.metalnessMapUv),y.push(b.roughnessMapUv),y.push(b.anisotropyMapUv),y.push(b.clearcoatMapUv),y.push(b.clearcoatNormalMapUv),y.push(b.clearcoatRoughnessMapUv),y.push(b.iridescenceMapUv),y.push(b.iridescenceThicknessMapUv),y.push(b.sheenColorMapUv),y.push(b.sheenRoughnessMapUv),y.push(b.specularMapUv),y.push(b.specularColorMapUv),y.push(b.specularIntensityMapUv),y.push(b.transmissionMapUv),y.push(b.thicknessMapUv),y.push(b.combine),y.push(b.fogExp2),y.push(b.sizeAttenuation),y.push(b.morphTargetsCount),y.push(b.morphAttributeCount),y.push(b.numSunLights),y.push(b.numDirLights),y.push(b.numPointLights),y.push(b.numSpotLights),y.push(b.numSpotLightMaps),y.push(b.numHemiLights),y.push(b.numRectAreaLights),y.push(b.numSunLightShadows),y.push(b.numDirLightShadows),y.push(b.numPointLightShadows),y.push(b.numSpotLightShadows),y.push(b.numSpotLightShadowsWithMaps),y.push(b.numLightProbes),y.push(b.shadowMapType),y.push(b.toneMapping),y.push(b.numClippingPlanes),y.push(b.numClipIntersection),y.push(b.depthPacking)}function g(y,b){a.disableAll(),b.instancing&&a.enable(0),b.instancingColor&&a.enable(1),b.instancingMorph&&a.enable(2),b.matcap&&a.enable(3),b.envMap&&a.enable(4),b.normalMapObjectSpace&&a.enable(5),b.normalMapTangentSpace&&a.enable(6),b.clearcoat&&a.enable(7),b.iridescence&&a.enable(8),b.alphaTest&&a.enable(9),b.vertexColors&&a.enable(10),b.vertexAlphas&&a.enable(11),b.vertexUv1s&&a.enable(12),b.vertexUv2s&&a.enable(13),b.vertexUv3s&&a.enable(14),b.vertexTangents&&a.enable(15),b.anisotropy&&a.enable(16),b.alphaHash&&a.enable(17),b.batching&&a.enable(18),b.dispersion&&a.enable(19),b.retroreflection&&a.enable(24),b.batchingColor&&a.enable(20),b.gradientMap&&a.enable(21),b.packedNormalMap&&a.enable(22),b.vertexNormals&&a.enable(23),y.push(a.mask),a.disableAll(),b.fog&&a.enable(0),b.useFog&&a.enable(1),b.flatShading&&a.enable(2),b.logarithmicDepthBuffer&&a.enable(3),b.reversedDepthBuffer&&a.enable(4),b.skinning&&a.enable(5),b.morphTargets&&a.enable(6),b.morphNormals&&a.enable(7),b.morphColors&&a.enable(8),b.premultipliedAlpha&&a.enable(9),b.shadowMapEnabled&&a.enable(10),b.doubleSided&&a.enable(11),b.flipSided&&a.enable(12),b.useDepthPacking&&a.enable(13),b.dithering&&a.enable(14),b.transmission&&a.enable(15),b.sheen&&a.enable(16),b.opaque&&a.enable(17),b.pointsUvs&&a.enable(18),b.decodeVideoTexture&&a.enable(19),b.decodeVideoTextureEmissive&&a.enable(20),b.alphaToCoverage&&a.enable(21),b.numLightProbeGrids>0&&a.enable(22),b.hasPositionAttribute&&a.enable(23),y.push(a.mask)}function S(y){let b=d[y.type],R;if(b){let N=xs[b];R=H1.clone(N.uniforms)}else R=y.uniforms;return R}function _(y,b){let R=h.get(b);return R!==void 0?++R.usedTimes:(R=new uU(t,b,y,s),c.push(R),h.set(b,R)),R}function E(y){if(--y.usedTimes===0){let b=c.indexOf(y);c[b]=c[c.length-1],c.pop(),h.delete(y.cacheKey),y.destroy()}}function T(y){o.remove(y)}function C(){o.dispose()}return{getParameters:M,getProgramCacheKey:m,getUniforms:S,acquireProgram:_,releaseProgram:E,releaseShaderCache:T,programs:c,dispose:C}}function pU(){let t=new WeakMap;function e(a){return t.has(a)}function n(a){let o=t.get(a);return o===void 0&&(o={},t.set(a,o)),o}function i(a){t.delete(a)}function s(a,o,l){t.get(a)[o]=l}function r(){t=new WeakMap}return{has:e,get:n,remove:i,update:s,dispose:r}}function mU(t,e){return t.groupOrder!==e.groupOrder?t.groupOrder-e.groupOrder:t.renderOrder!==e.renderOrder?t.renderOrder-e.renderOrder:t.material.id!==e.material.id?t.material.id-e.material.id:t.materialVariant!==e.materialVariant?t.materialVariant-e.materialVariant:t.z!==e.z?t.z-e.z:t.id-e.id}function aM(t,e){return t.groupOrder!==e.groupOrder?t.groupOrder-e.groupOrder:t.renderOrder!==e.renderOrder?t.renderOrder-e.renderOrder:t.z!==e.z?e.z-t.z:t.id-e.id}function oM(){let t=[],e=0,n=[],i=[],s=[];function r(){e=0,n.length=0,i.length=0,s.length=0}function a(u){let d=0;return u.isInstancedMesh&&(d+=2),u.isSkinnedMesh&&(d+=1),d}function o(u,d,v,M,m,f){let g=t[e];return g===void 0?(g={id:u.id,object:u,geometry:d,material:v,materialVariant:a(u),groupOrder:M,renderOrder:u.renderOrder,z:m,group:f},t[e]=g):(g.id=u.id,g.object=u,g.geometry=d,g.material=v,g.materialVariant=a(u),g.groupOrder=M,g.renderOrder=u.renderOrder,g.z=m,g.group=f),e++,g}function l(u,d,v,M,m,f,g){g.reversedDepth===!0&&(m=-m);let S=o(u,d,v,M,m,f);v.transmission>0?i.push(S):v.transparent===!0?s.push(S):n.push(S)}function c(u,d,v,M,m,f){let g=o(u,d,v,M,m,f);v.transmission>0?i.unshift(g):v.transparent===!0?s.unshift(g):n.unshift(g)}function h(u,d){n.length>1&&n.sort(u||mU),i.length>1&&i.sort(d||aM),s.length>1&&s.sort(d||aM)}function p(){for(let u=e,d=t.length;u<d;u++){let v=t[u];if(v.id===null)break;v.id=null,v.object=null,v.geometry=null,v.material=null,v.group=null}}return{opaque:n,transmissive:i,transparent:s,init:r,push:l,unshift:c,finish:p,sort:h}}function gU(){let t=new WeakMap;function e(i,s){let r=t.get(i),a;return r===void 0?(a=new oM,t.set(i,[a])):s>=r.length?(a=new oM,r.push(a)):a=r[s],a}function n(){t=new WeakMap}return{get:e,dispose:n}}function vU(){let t={};return{get:function(e){if(t[e.id]!==void 0)return t[e.id];let n;switch(e.type){case"SunLight":case"DirectionalLight":n={direction:new G,color:new Ye};break;case"SpotLight":n={position:new G,direction:new G,color:new Ye,distance:0,coneCos:0,penumbraCos:0,decay:0};break;case"PointLight":n={position:new G,color:new Ye,distance:0,decay:0};break;case"HemisphereLight":n={direction:new G,skyColor:new Ye,groundColor:new Ye};break;case"RectAreaLight":n={color:new Ye,position:new G,halfWidth:new G,halfHeight:new G};break}return t[e.id]=n,n}}}function xU(){let t={};return{get:function(e){if(t[e.id]!==void 0)return t[e.id];let n;switch(e.type){case"SunLight":case"DirectionalLight":n={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Ne};break;case"SpotLight":n={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Ne};break;case"PointLight":n={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Ne,shadowCameraNear:1,shadowCameraFar:1e3};break}return t[e.id]=n,n}}}var yU=0;function _U(t,e){return(e.castShadow?2:0)-(t.castShadow?2:0)+(e.map?1:0)-(t.map?1:0)}function SU(t){let e=new vU,n=xU(),i={version:0,hash:{sunLength:-1,directionalLength:-1,pointLength:-1,spotLength:-1,rectAreaLength:-1,hemiLength:-1,numSunShadows:-1,numDirectionalShadows:-1,numPointShadows:-1,numSpotShadows:-1,numSpotMaps:-1,numLightProbes:-1},ambient:[0,0,0],probe:[],sun:[],sunShadow:[],sunShadowMap:[],sunShadowMatrix:[],sunShadowCascade:[],directional:[],directionalShadow:[],directionalShadowMap:[],directionalShadowMatrix:[],spot:[],spotLightMap:[],spotShadow:[],spotShadowMap:[],spotLightMatrix:[],rectArea:[],rectAreaLTC1:null,rectAreaLTC2:null,point:[],pointShadow:[],pointShadowMap:[],pointShadowMatrix:[],hemi:[],numSpotLightShadowsWithMaps:0,numLightProbes:0};for(let c=0;c<9;c++)i.probe.push(new G);let s=new G,r=new kt,a=new kt;function o(c){let h=0,p=0,u=0;for(let F=0;F<9;F++)i.probe[F].set(0,0,0);let d=0,v=0,M=0,m=0,f=0,g=0,S=0,_=0,E=0,T=0,C=0,y=0,b=0,R=0;c.sort(_U);for(let F=0,k=c.length;F<k;F++){let B=c[F],z=B.color,Z=B.intensity,q=B.distance,ie=null;if(B.shadow&&B.shadow.map&&(B.shadow.map.texture.format===Fr?ie=B.shadow.map.texture:ie=B.shadow.map.depthTexture||B.shadow.map.texture),B.isAmbientLight)h+=z.r*Z,p+=z.g*Z,u+=z.b*Z;else if(B.isLightProbe){for(let W=0;W<9;W++)i.probe[W].addScaledVector(B.sh.coefficients[W],Z);R++}else if(B.isSunLight){let W=e.get(B);if(W.color.copy(B.color).multiplyScalar(B.intensity),B.castShadow){let $=B.shadow,te=n.get(B);te.shadowIntensity=$.intensity,te.shadowBias=$.bias,te.shadowNormalBias=$.normalBias,te.shadowRadius=$.radius,te.shadowMapSize.copy($.mapSize).multiply($.getFrameExtents()),i.sunShadow[v]=te,i.sunShadowMap[v]=ie;let we=$.getViewportCount();for(let Me=0;Me<we;Me++)i.sunShadowMatrix[M+Me]=$.getMatrix(Me),i.sunShadowCascade[M+Me]=$._cascadeData[Me];M+=we,v++}i.sun[d]=W,d++}else if(B.isDirectionalLight){let W=e.get(B);if(W.color.copy(B.color).multiplyScalar(B.intensity),B.castShadow){let $=B.shadow,te=n.get(B);te.shadowIntensity=$.intensity,te.shadowBias=$.bias,te.shadowNormalBias=$.normalBias,te.shadowRadius=$.radius,te.shadowMapSize=$.mapSize,i.directionalShadow[m]=te,i.directionalShadowMap[m]=ie,i.directionalShadowMatrix[m]=B.shadow.matrix,E++}i.directional[m]=W,m++}else if(B.isSpotLight){let W=e.get(B);W.position.setFromMatrixPosition(B.matrixWorld),W.color.copy(z).multiplyScalar(Z),W.distance=q,W.coneCos=Math.cos(B.angle),W.penumbraCos=Math.cos(B.angle*(1-B.penumbra)),W.decay=B.decay,i.spot[g]=W;let $=B.shadow;if(B.map&&(i.spotLightMap[y]=B.map,y++,$.updateMatrices(B),B.castShadow&&b++),i.spotLightMatrix[g]=$.matrix,B.castShadow){let te=n.get(B);te.shadowIntensity=$.intensity,te.shadowBias=$.bias,te.shadowNormalBias=$.normalBias,te.shadowRadius=$.radius,te.shadowMapSize=$.mapSize,i.spotShadow[g]=te,i.spotShadowMap[g]=ie,C++}g++}else if(B.isRectAreaLight){let W=e.get(B);W.color.copy(z).multiplyScalar(Z),W.halfWidth.set(B.width*.5,0,0),W.halfHeight.set(0,B.height*.5,0),i.rectArea[S]=W,S++}else if(B.isPointLight){let W=e.get(B);if(W.color.copy(B.color).multiplyScalar(B.intensity),W.distance=B.distance,W.decay=B.decay,B.castShadow){let $=B.shadow,te=n.get(B);te.shadowIntensity=$.intensity,te.shadowBias=$.bias,te.shadowNormalBias=$.normalBias,te.shadowRadius=$.radius,te.shadowMapSize=$.mapSize,te.shadowCameraNear=$.camera.near,te.shadowCameraFar=$.camera.far,i.pointShadow[f]=te,i.pointShadowMap[f]=ie,i.pointShadowMatrix[f]=B.shadow.matrix,T++}i.point[f]=W,f++}else if(B.isHemisphereLight){let W=e.get(B);W.skyColor.copy(B.color).multiplyScalar(Z),W.groundColor.copy(B.groundColor).multiplyScalar(Z),i.hemi[_]=W,_++}}S>0&&(t.has("OES_texture_float_linear")===!0?(i.rectAreaLTC1=he.LTC_FLOAT_1,i.rectAreaLTC2=he.LTC_FLOAT_2):(i.rectAreaLTC1=he.LTC_HALF_1,i.rectAreaLTC2=he.LTC_HALF_2)),i.ambient[0]=h,i.ambient[1]=p,i.ambient[2]=u;let N=i.hash;(N.sunLength!==d||N.directionalLength!==m||N.pointLength!==f||N.spotLength!==g||N.rectAreaLength!==S||N.hemiLength!==_||N.numSunShadows!==v||N.numDirectionalShadows!==E||N.numPointShadows!==T||N.numSpotShadows!==C||N.numSpotMaps!==y||N.numLightProbes!==R)&&(i.sun.length=d,i.directional.length=m,i.spot.length=g,i.rectArea.length=S,i.point.length=f,i.hemi.length=_,i.sunShadow.length=v,i.sunShadowMap.length=v,i.sunShadowMatrix.length=M,i.sunShadowCascade.length=M,i.directionalShadow.length=E,i.directionalShadowMap.length=E,i.directionalShadowMatrix.length=E,i.pointShadow.length=T,i.pointShadowMap.length=T,i.pointShadowMatrix.length=T,i.spotShadow.length=C,i.spotShadowMap.length=C,i.spotLightMatrix.length=C+y-b,i.spotLightMap.length=y,i.numSpotLightShadowsWithMaps=b,i.numLightProbes=R,N.sunLength=d,N.directionalLength=m,N.pointLength=f,N.spotLength=g,N.rectAreaLength=S,N.hemiLength=_,N.numSunShadows=v,N.numDirectionalShadows=E,N.numPointShadows=T,N.numSpotShadows=C,N.numSpotMaps=y,N.numLightProbes=R,i.version=yU++)}function l(c,h){let p=0,u=0,d=0,v=0,M=0,m=0,f=h.matrixWorldInverse;for(let g=0,S=c.length;g<S;g++){let _=c[g];if(_.isSunLight){let E=i.sun[p];E.direction.setFromMatrixPosition(_.matrixWorld),E.direction.transformDirection(f),p++}else if(_.isDirectionalLight){let E=i.directional[u];E.direction.setFromMatrixPosition(_.matrixWorld),s.setFromMatrixPosition(_.target.matrixWorld),E.direction.sub(s),E.direction.transformDirection(f),u++}else if(_.isSpotLight){let E=i.spot[v];E.position.setFromMatrixPosition(_.matrixWorld),E.position.applyMatrix4(f),E.direction.setFromMatrixPosition(_.matrixWorld),s.setFromMatrixPosition(_.target.matrixWorld),E.direction.sub(s),E.direction.transformDirection(f),v++}else if(_.isRectAreaLight){let E=i.rectArea[M];E.position.setFromMatrixPosition(_.matrixWorld),E.position.applyMatrix4(f),a.identity(),r.copy(_.matrixWorld),r.premultiply(f),a.extractRotation(r),E.halfWidth.set(_.width*.5,0,0),E.halfHeight.set(0,_.height*.5,0),E.halfWidth.applyMatrix4(a),E.halfHeight.applyMatrix4(a),M++}else if(_.isPointLight){let E=i.point[d];E.position.setFromMatrixPosition(_.matrixWorld),E.position.applyMatrix4(f),d++}else if(_.isHemisphereLight){let E=i.hemi[m];E.direction.setFromMatrixPosition(_.matrixWorld),E.direction.transformDirection(f),m++}}}return{setup:o,setupView:l,state:i}}function lM(t){let e=new SU(t),n=[],i=[],s=[];function r(u){p.camera=u,n.length=0,i.length=0,s.length=0}function a(u){n.push(u)}function o(u){i.push(u)}function l(u){s.push(u)}function c(){e.setup(n)}function h(u){e.setupView(n,u)}let p={lightsArray:n,shadowsArray:i,lightProbeGridArray:s,camera:null,lights:e,transmissionRenderTarget:{},textureUnits:0};return{init:r,state:p,setupLights:c,setupLightsView:h,pushLight:a,pushShadow:o,pushLightProbeGrid:l}}function AU(t){let e=new WeakMap;function n(s,r=0){let a=e.get(s),o;return a===void 0?(o=new lM(t),e.set(s,[o])):r>=a.length?(o=new lM(t),a.push(o)):o=a[r],o}function i(){e=new WeakMap}return{get:n,dispose:i}}var MU=`void main() {
	gl_Position = vec4( position, 1.0 );
}`,EU=`uniform sampler2D shadow_pass;
uniform vec2 resolution;
uniform float radius;
void main() {
	const float samples = float( VSM_SAMPLES );
	float mean = 0.0;
	float squared_mean = 0.0;
	float uvStride = samples <= 1.0 ? 0.0 : 2.0 / ( samples - 1.0 );
	float uvStart = samples <= 1.0 ? 0.0 : - 1.0;
	for ( float i = 0.0; i < samples; i ++ ) {
		float uvOffset = uvStart + i * uvStride;
		#ifdef HORIZONTAL_PASS
			vec2 distribution = texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( uvOffset, 0.0 ) * radius ) / resolution ).rg;
			mean += distribution.x;
			squared_mean += distribution.y * distribution.y + distribution.x * distribution.x;
		#else
			float depth = texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( 0.0, uvOffset ) * radius ) / resolution ).r;
			mean += depth;
			squared_mean += depth * depth;
		#endif
	}
	mean = mean / samples;
	squared_mean = squared_mean / samples;
	float std_dev = sqrt( max( 0.0, squared_mean - mean * mean ) );
	gl_FragColor = vec4( mean, std_dev, 0.0, 1.0 );
}`,TU=[new G(1,0,0),new G(-1,0,0),new G(0,1,0),new G(0,-1,0),new G(0,0,1),new G(0,0,-1)],bU=[new G(0,-1,0),new G(0,-1,0),new G(0,0,1),new G(0,0,-1),new G(0,-1,0),new G(0,-1,0)],cM=new kt,Lc=new G,V0=new G;function wU(t,e,n){let i=new xc,s=new Ne,r=new Ne,a=new Ot,o=new hh,l=new dh,c={},h=n.maxTextureSize,p={[ps]:$t,[$t]:ps,[zn]:zn},u=new Wt({defines:{VSM_SAMPLES:8},uniforms:{shadow_pass:{value:null},resolution:{value:new Ne},radius:{value:4}},vertexShader:MU,fragmentShader:EU}),d=u.clone();d.defines.HORIZONTAL_PASS=1;let v=new wi;v.setAttribute("position",new Fn(new Float32Array([-1,-1,.5,3,-1,.5,-1,3,.5]),3));let M=new Rn(v,u),m=this;this.enabled=!1,this.autoUpdate=!0,this.needsUpdate=!1,this.type=Ec;let f=this.type;this.render=function(T,C,y){if(m.enabled===!1||m.autoUpdate===!1&&m.needsUpdate===!1||T.length===0)return;this.type===s1&&(De("WebGLShadowMap: PCFSoftShadowMap has been removed. Using PCFShadowMap instead."),this.type=Ec);let b=t.getRenderTarget(),R=t.getActiveCubeFace(),N=t.getActiveMipmapLevel(),F=t.state;F.setBlending(Kn),F.buffers.depth.getReversed()===!0?F.buffers.color.setClear(0,0,0,0):F.buffers.color.setClear(1,1,1,1),F.buffers.depth.setTest(!0),F.setScissorTest(!1);let k=f!==this.type;k&&C.traverse(function(B){B.material&&(Array.isArray(B.material)?B.material.forEach(z=>z.needsUpdate=!0):B.material.needsUpdate=!0)});for(let B=0,z=T.length;B<z;B++){let Z=T[B],q=Z.shadow;if(q===void 0){De("WebGLShadowMap:",Z,"has no shadow.");continue}if(q.autoUpdate===!1&&q.needsUpdate===!1)continue;s.copy(q.mapSize);let ie=q.getFrameExtents();s.multiply(ie),r.copy(q.mapSize),(s.x>h||s.y>h)&&(s.x>h&&(r.x=Math.floor(h/ie.x),s.x=r.x*ie.x,q.mapSize.x=r.x),s.y>h&&(r.y=Math.floor(h/ie.y),s.y=r.y*ie.y,q.mapSize.y=r.y));let W=t.state.buffers.depth.getReversed();if(q.camera._reversedDepth=W,q.map===null||k===!0){if(q.map!==null&&(q.map.depthTexture!==null&&(q.map.depthTexture.dispose(),q.map.depthTexture=null),q.map.dispose()),this.type===Qo){if(Z.isPointLight){De("WebGLShadowMap: VSM shadow maps are not supported for PointLights. Use PCF or BasicShadowMap instead.");continue}q.map=new Ft(s.x,s.y,{format:Fr,type:Xi,minFilter:wt,magFilter:wt,generateMipmaps:!1}),q.map.texture.name=Z.name+".shadowMap",q.map.depthTexture=new Ci(s.x,s.y,fi),q.map.depthTexture.name=Z.name+".shadowMapDepth",q.map.depthTexture.format=hs,q.map.depthTexture.compareFunction=null,q.map.depthTexture.minFilter=dn,q.map.depthTexture.magFilter=dn}else Z.isPointLight?(q.map=new vd(s.x),q.map.depthTexture=new uh(s.x,Wi)):(q.map=new Ft(s.x,s.y),q.map.depthTexture=new Ci(s.x,s.y,Wi)),q.map.depthTexture.name=Z.name+".shadowMap",q.map.depthTexture.format=hs,this.type===Ec?(q.map.depthTexture.compareFunction=W?dd:hd,q.map.depthTexture.minFilter=wt,q.map.depthTexture.magFilter=wt):(q.map.depthTexture.compareFunction=null,q.map.depthTexture.minFilter=dn,q.map.depthTexture.magFilter=dn);q.camera.updateProjectionMatrix()}q.map.isWebGLCubeRenderTarget!==!0&&(q.map.width!==s.x||q.map.height!==s.y)&&q.map.setSize(s.x,s.y);let $=q.map.isWebGLCubeRenderTarget?6:q.getViewportCount();Z.isPointLight!==!0&&q.updateMatrices(Z,y);for(let te=0;te<$;te++){let we=q.getCamera(te);if(Z.isPointLight){let Me=q.camera,ut=q.matrix,qe=Z.distance||Me.far;qe!==Me.far&&(Me.far=qe,Me.updateProjectionMatrix()),Lc.setFromMatrixPosition(Z.matrixWorld),Me.position.copy(Lc),V0.copy(Me.position),V0.add(TU[te]),Me.up.copy(bU[te]),Me.lookAt(V0),Me.updateMatrixWorld(),ut.makeTranslation(-Lc.x,-Lc.y,-Lc.z),cM.multiplyMatrices(Me.projectionMatrix,Me.matrixWorldInverse),q._frustum.setFromProjectionMatrix(cM,Me.coordinateSystem,Me.reversedDepth)}if(q.map.isWebGLCubeRenderTarget)t.setRenderTarget(q.map,te),t.clear();else{te===0&&(t.setRenderTarget(q.map),t.clear());let Me=q.getViewport(te);a.set(r.x*Me.x,r.y*Me.y,r.x*Me.z,r.y*Me.w),F.viewport(a)}i=q.getFrustum(te),_(C,y,we,Z,this.type)}q.isPointLightShadow!==!0&&this.type===Qo&&g(q,y),q.needsUpdate=!1}f=this.type,m.needsUpdate=!1,t.setRenderTarget(b,R,N)};function g(T,C){let y=e.update(M);u.defines.VSM_SAMPLES!==T.blurSamples&&(u.defines.VSM_SAMPLES=T.blurSamples,d.defines.VSM_SAMPLES=T.blurSamples,u.needsUpdate=!0,d.needsUpdate=!0),T.mapPass===null?T.mapPass=new Ft(s.x,s.y,{format:Fr,type:Xi}):(T.mapPass.width!==T.map.width||T.mapPass.height!==T.map.height)&&T.mapPass.setSize(T.map.width,T.map.height),u.uniforms.shadow_pass.value=T.map.depthTexture,u.uniforms.resolution.value.set(T.map.width,T.map.height),u.uniforms.radius.value=T.radius,t.setRenderTarget(T.mapPass),t.clear(),t.renderBufferDirect(C,null,y,u,M,null),d.uniforms.shadow_pass.value=T.mapPass.texture,d.uniforms.resolution.value.set(T.map.width,T.map.height),d.uniforms.radius.value=T.radius,t.setRenderTarget(T.map),t.clear(),t.renderBufferDirect(C,null,y,d,M,null)}function S(T,C,y,b){let R=null,N=y.isPointLight===!0?T.customDistanceMaterial:T.customDepthMaterial;if(N!==void 0)R=N;else if(R=y.isPointLight===!0?l:o,t.localClippingEnabled&&C.clipShadows===!0&&Array.isArray(C.clippingPlanes)&&C.clippingPlanes.length!==0||C.displacementMap&&C.displacementScale!==0||C.alphaMap&&C.alphaTest>0||C.map&&C.alphaTest>0||C.alphaToCoverage===!0){let F=R.uuid,k=C.uuid,B=c[F];B===void 0&&(B={},c[F]=B);let z=B[k];z===void 0&&(z=R.clone(),B[k]=z,C.addEventListener("dispose",E)),R=z}if(R.visible=C.visible,R.wireframe=C.wireframe,b===Qo?R.side=C.shadowSide!==null?C.shadowSide:C.side:R.side=C.shadowSide!==null?C.shadowSide:p[C.side],R.alphaMap=C.alphaMap,R.alphaTest=C.alphaToCoverage===!0?.5:C.alphaTest,R.map=C.map,R.clipShadows=C.clipShadows,R.clippingPlanes=C.clippingPlanes,R.clipIntersection=C.clipIntersection,R.displacementMap=C.displacementMap,R.displacementScale=C.displacementScale,R.displacementBias=C.displacementBias,R.wireframeLinewidth=C.wireframeLinewidth,R.linewidth=C.linewidth,y.isPointLight===!0&&R.isMeshDistanceMaterial===!0){let F=t.properties.get(R);F.light=y}return R}function _(T,C,y,b,R){if(T.visible===!1)return;if(T.layers.test(C.layers)&&(T.isMesh||T.isLine||T.isPoints)&&(T.castShadow||T.receiveShadow&&R===Qo)&&(!T.frustumCulled||T.intersectsFrustum(i))){T.modelViewMatrix.multiplyMatrices(y.matrixWorldInverse,T.matrixWorld);let k=e.update(T),B=T.material;if(Array.isArray(B)){let z=k.groups;for(let Z=0,q=z.length;Z<q;Z++){let ie=z[Z],W=B[ie.materialIndex];if(W&&W.visible){let $=S(T,W,b,R);T.onBeforeShadow(t,T,C,y,k,$,ie),t.renderBufferDirect(y,null,k,$,T,ie),T.onAfterShadow(t,T,C,y,k,$,ie)}}}else if(B.visible){let z=S(T,B,b,R);T.onBeforeShadow(t,T,C,y,k,z,null),t.renderBufferDirect(y,null,k,z,T,null),T.onAfterShadow(t,T,C,y,k,z,null)}}let F=T.children;for(let k=0,B=F.length;k<B;k++)_(F[k],C,y,b,R)}function E(T){T.target.removeEventListener("dispose",E);for(let y in c){let b=c[y],R=T.target.uuid;R in b&&(b[R].dispose(),delete b[R])}}}function CU(t,e){function n(){let U=!1,le=new Ot,K=null,ce=new Ot(0,0,0,0);return{setMask:function(me){K!==me&&!U&&(t.colorMask(me,me,me,me),K=me)},setLocked:function(me){U=me},setClear:function(me,ne,Ce,Ae,St){St===!0&&(me*=Ae,ne*=Ae,Ce*=Ae),le.set(me,ne,Ce,Ae),ce.equals(le)===!1&&(t.clearColor(me,ne,Ce,Ae),ce.copy(le))},reset:function(){U=!1,K=null,ce.set(-1,0,0,0)}}}function i(){let U=!1,le=!1,K=null,ce=null,me=null;return{setReversed:function(ne){if(le!==ne){let Ce=e.get("EXT_clip_control");ne?Ce.clipControlEXT(Ce.LOWER_LEFT_EXT,Ce.ZERO_TO_ONE_EXT):Ce.clipControlEXT(Ce.LOWER_LEFT_EXT,Ce.NEGATIVE_ONE_TO_ONE_EXT),le=ne;let Ae=me;me=null,this.setClear(Ae)}},getReversed:function(){return le},setTest:function(ne){ne?ee(t.DEPTH_TEST):ue(t.DEPTH_TEST)},setMask:function(ne){K!==ne&&!U&&(t.depthMask(ne),K=ne)},setFunc:function(ne){if(le&&(ne=F1[ne]),ce!==ne){switch(ne){case Zf:t.depthFunc(t.NEVER);break;case Go:t.depthFunc(t.ALWAYS);break;case Kf:t.depthFunc(t.LESS);break;case Vo:t.depthFunc(t.LEQUAL);break;case Jf:t.depthFunc(t.EQUAL);break;case jf:t.depthFunc(t.GEQUAL);break;case $f:t.depthFunc(t.GREATER);break;case eh:t.depthFunc(t.NOTEQUAL);break;default:t.depthFunc(t.LEQUAL)}ce=ne}},setLocked:function(ne){U=ne},setClear:function(ne){me!==ne&&(me=ne,le&&(ne=1-ne),t.clearDepth(ne))},reset:function(){U=!1,K=null,ce=null,me=null,le=!1}}}function s(){let U=!1,le=null,K=null,ce=null,me=null,ne=null,Ce=null,Ae=null,St=null;return{setTest:function(at){U||(at?ee(t.STENCIL_TEST):ue(t.STENCIL_TEST))},setMask:function(at){le!==at&&!U&&(t.stencilMask(at),le=at)},setFunc:function(at,Di,Yi){(K!==at||ce!==Di||me!==Yi)&&(t.stencilFunc(at,Di,Yi),K=at,ce=Di,me=Yi)},setOp:function(at,Di,Yi){(ne!==at||Ce!==Di||Ae!==Yi)&&(t.stencilOp(at,Di,Yi),ne=at,Ce=Di,Ae=Yi)},setLocked:function(at){U=at},setClear:function(at){St!==at&&(t.clearStencil(at),St=at)},reset:function(){U=!1,le=null,K=null,ce=null,me=null,ne=null,Ce=null,Ae=null,St=null}}}let r=new n,a=new i,o=new s,l=new WeakMap,c=new WeakMap,h={},p={},u={},d=new WeakMap,v=[],M=null,m=!1,f=null,g=null,S=null,_=null,E=null,T=null,C=null,y=new Ye(0,0,0),b=0,R=!1,N=null,F=null,k=null,B=null,z=null,Z=t.getParameter(t.MAX_COMBINED_TEXTURE_IMAGE_UNITS),q=!1,ie=0,W=t.getParameter(t.VERSION);W.indexOf("WebGL")!==-1?(ie=parseFloat(/^WebGL (\d)/.exec(W)[1]),q=ie>=1):W.indexOf("OpenGL ES")!==-1&&(ie=parseFloat(/^OpenGL ES (\d)/.exec(W)[1]),q=ie>=2);let $=null,te={},we=t.getParameter(t.SCISSOR_BOX),Me=t.getParameter(t.VIEWPORT),ut=new Ot().fromArray(we),qe=new Ot().fromArray(Me);function $e(U,le,K,ce){let me=new Uint8Array(4),ne=t.createTexture();t.bindTexture(U,ne),t.texParameteri(U,t.TEXTURE_MIN_FILTER,t.NEAREST),t.texParameteri(U,t.TEXTURE_MAG_FILTER,t.NEAREST);for(let Ce=0;Ce<K;Ce++)U===t.TEXTURE_3D||U===t.TEXTURE_2D_ARRAY?t.texImage3D(le,0,t.RGBA,1,1,ce,0,t.RGBA,t.UNSIGNED_BYTE,me):t.texImage2D(le+Ce,0,t.RGBA,1,1,0,t.RGBA,t.UNSIGNED_BYTE,me);return ne}let X={};X[t.TEXTURE_2D]=$e(t.TEXTURE_2D,t.TEXTURE_2D,1),X[t.TEXTURE_CUBE_MAP]=$e(t.TEXTURE_CUBE_MAP,t.TEXTURE_CUBE_MAP_POSITIVE_X,6),X[t.TEXTURE_2D_ARRAY]=$e(t.TEXTURE_2D_ARRAY,t.TEXTURE_2D_ARRAY,1,1),X[t.TEXTURE_3D]=$e(t.TEXTURE_3D,t.TEXTURE_3D,1,1),r.setClear(0,0,0,1),a.setClear(1),o.setClear(0),ee(t.DEPTH_TEST),a.setFunc(Vo),Qe(!1),Ut(s0),ee(t.CULL_FACE),tt(Kn);function ee(U){h[U]!==!0&&(t.enable(U),h[U]=!0)}function ue(U){h[U]!==!1&&(t.disable(U),h[U]=!1)}function Se(U,le){return u[U]!==le?(t.bindFramebuffer(U,le),u[U]=le,U===t.DRAW_FRAMEBUFFER&&(u[t.FRAMEBUFFER]=le),U===t.FRAMEBUFFER&&(u[t.DRAW_FRAMEBUFFER]=le),!0):!1}function ge(U,le){let K=v,ce=!1;if(U){K=d.get(le),K===void 0&&(K=[],d.set(le,K));let me=U.textures;if(K.length!==me.length||K[0]!==t.COLOR_ATTACHMENT0){for(let ne=0,Ce=me.length;ne<Ce;ne++)K[ne]=t.COLOR_ATTACHMENT0+ne;K.length=me.length,ce=!0}}else K[0]!==t.BACK&&(K[0]=t.BACK,ce=!0);ce&&t.drawBuffers(K)}function Fe(U){return M!==U?(t.useProgram(U),M=U,!0):!1}let en={[_a]:t.FUNC_ADD,[a1]:t.FUNC_SUBTRACT,[o1]:t.FUNC_REVERSE_SUBTRACT};en[l1]=t.MIN,en[c1]=t.MAX;let Ve={[u1]:t.ZERO,[f1]:t.ONE,[h1]:t.SRC_COLOR,[l0]:t.SRC_ALPHA,[x1]:t.SRC_ALPHA_SATURATE,[g1]:t.DST_COLOR,[p1]:t.DST_ALPHA,[d1]:t.ONE_MINUS_SRC_COLOR,[c0]:t.ONE_MINUS_SRC_ALPHA,[v1]:t.ONE_MINUS_DST_COLOR,[m1]:t.ONE_MINUS_DST_ALPHA,[y1]:t.CONSTANT_COLOR,[_1]:t.ONE_MINUS_CONSTANT_COLOR,[S1]:t.CONSTANT_ALPHA,[A1]:t.ONE_MINUS_CONSTANT_ALPHA};function tt(U,le,K,ce,me,ne,Ce,Ae,St,at){if(U===Kn){m===!0&&(ue(t.BLEND),m=!1);return}if(m===!1&&(ee(t.BLEND),m=!0),U!==r1){if(U!==f||at!==R){if((g!==_a||E!==_a)&&(t.blendEquation(t.FUNC_ADD),g=_a,E=_a),at)switch(U){case Zo:t.blendFuncSeparate(t.ONE,t.ONE_MINUS_SRC_ALPHA,t.ONE,t.ONE_MINUS_SRC_ALPHA);break;case r0:t.blendFunc(t.ONE,t.ONE);break;case a0:t.blendFuncSeparate(t.ZERO,t.ONE_MINUS_SRC_COLOR,t.ZERO,t.ONE);break;case o0:t.blendFuncSeparate(t.DST_COLOR,t.ONE_MINUS_SRC_ALPHA,t.ZERO,t.ONE);break;default:Ie("WebGLState: Invalid blending: ",U);break}else switch(U){case Zo:t.blendFuncSeparate(t.SRC_ALPHA,t.ONE_MINUS_SRC_ALPHA,t.ONE,t.ONE_MINUS_SRC_ALPHA);break;case r0:t.blendFuncSeparate(t.SRC_ALPHA,t.ONE,t.ONE,t.ONE);break;case a0:Ie("WebGLState: SubtractiveBlending requires material.premultipliedAlpha = true");break;case o0:Ie("WebGLState: MultiplyBlending requires material.premultipliedAlpha = true");break;default:Ie("WebGLState: Invalid blending: ",U);break}S=null,_=null,T=null,C=null,y.set(0,0,0),b=0,f=U,R=at}return}me=me||le,ne=ne||K,Ce=Ce||ce,(le!==g||me!==E)&&(t.blendEquationSeparate(en[le],en[me]),g=le,E=me),(K!==S||ce!==_||ne!==T||Ce!==C)&&(t.blendFuncSeparate(Ve[K],Ve[ce],Ve[ne],Ve[Ce]),S=K,_=ce,T=ne,C=Ce),(Ae.equals(y)===!1||St!==b)&&(t.blendColor(Ae.r,Ae.g,Ae.b,St),y.copy(Ae),b=St),f=U,R=!1}function _t(U,le){U.side===zn?ue(t.CULL_FACE):ee(t.CULL_FACE);let K=U.side===$t;le&&(K=!K),Qe(K),U.blending===Zo&&U.transparent===!1?tt(Kn):tt(U.blending,U.blendEquation,U.blendSrc,U.blendDst,U.blendEquationAlpha,U.blendSrcAlpha,U.blendDstAlpha,U.blendColor,U.blendAlpha,U.premultipliedAlpha),a.setFunc(U.depthFunc),a.setTest(U.depthTest),a.setMask(U.depthWrite),r.setMask(U.colorWrite);let ce=U.stencilWrite;o.setTest(ce),ce&&(o.setMask(U.stencilWriteMask),o.setFunc(U.stencilFunc,U.stencilRef,U.stencilFuncMask),o.setOp(U.stencilFail,U.stencilZFail,U.stencilZPass)),Hn(U.polygonOffset,U.polygonOffsetFactor,U.polygonOffsetUnits),U.alphaToCoverage===!0?ee(t.SAMPLE_ALPHA_TO_COVERAGE):ue(t.SAMPLE_ALPHA_TO_COVERAGE)}function Qe(U){N!==U&&(U?t.frontFace(t.CW):t.frontFace(t.CCW),N=U)}function Ut(U){U!==n1?(ee(t.CULL_FACE),U!==F&&(U===s0?t.cullFace(t.BACK):U===i1?t.cullFace(t.FRONT):t.cullFace(t.FRONT_AND_BACK))):ue(t.CULL_FACE),F=U}function on(U){U!==k&&(q&&t.lineWidth(U),k=U)}function Hn(U,le,K){U?(ee(t.POLYGON_OFFSET_FILL),(B!==le||z!==K)&&(B=le,z=K,a.getReversed()&&(le=-le),t.polygonOffset(le,K))):ue(t.POLYGON_OFFSET_FILL)}function Pt(U){U?ee(t.SCISSOR_TEST):ue(t.SCISSOR_TEST)}function Yt(U){U===void 0&&(U=t.TEXTURE0+Z-1),$!==U&&(t.activeTexture(U),$=U)}function I(U,le,K){K===void 0&&($===null?K=t.TEXTURE0+Z-1:K=$);let ce=te[K];ce===void 0&&(ce={type:void 0,texture:void 0},te[K]=ce),(ce.type!==U||ce.texture!==le)&&($!==K&&(t.activeTexture(K),$=K),t.bindTexture(U,le||X[U]),ce.type=U,ce.texture=le)}function Sn(){let U=te[$];U!==void 0&&U.type!==void 0&&(t.bindTexture(U.type,null),U.type=void 0,U.texture=void 0)}function ft(){try{t.compressedTexImage2D(...arguments)}catch(U){Ie("WebGLState:",U)}}function w(){try{t.compressedTexImage3D(...arguments)}catch(U){Ie("WebGLState:",U)}}function x(){try{t.texSubImage2D(...arguments)}catch(U){Ie("WebGLState:",U)}}function P(){try{t.texSubImage3D(...arguments)}catch(U){Ie("WebGLState:",U)}}function H(){try{t.compressedTexSubImage2D(...arguments)}catch(U){Ie("WebGLState:",U)}}function Y(){try{t.compressedTexSubImage3D(...arguments)}catch(U){Ie("WebGLState:",U)}}function se(){try{t.texStorage2D(...arguments)}catch(U){Ie("WebGLState:",U)}}function re(){try{t.texStorage3D(...arguments)}catch(U){Ie("WebGLState:",U)}}function Q(){try{t.texImage2D(...arguments)}catch(U){Ie("WebGLState:",U)}}function j(){try{t.texImage3D(...arguments)}catch(U){Ie("WebGLState:",U)}}function ae(U){return p[U]!==void 0?p[U]:t.getParameter(U)}function Te(U,le){p[U]!==le&&(t.pixelStorei(U,le),p[U]=le)}function fe(U){ut.equals(U)===!1&&(t.scissor(U.x,U.y,U.z,U.w),ut.copy(U))}function oe(U){qe.equals(U)===!1&&(t.viewport(U.x,U.y,U.z,U.w),qe.copy(U))}function be(U,le){let K=c.get(le);K===void 0&&(K=new WeakMap,c.set(le,K));let ce=K.get(U);ce===void 0&&(ce=t.getUniformBlockIndex(le,U.name),K.set(U,ce))}function Re(U,le){let ce=c.get(le).get(U);l.get(le)!==ce&&(t.uniformBlockBinding(le,ce,U.__bindingPointIndex),l.set(le,ce))}function Le(){t.disable(t.BLEND),t.disable(t.CULL_FACE),t.disable(t.DEPTH_TEST),t.disable(t.POLYGON_OFFSET_FILL),t.disable(t.SCISSOR_TEST),t.disable(t.STENCIL_TEST),t.disable(t.SAMPLE_ALPHA_TO_COVERAGE),t.blendEquation(t.FUNC_ADD),t.blendFunc(t.ONE,t.ZERO),t.blendFuncSeparate(t.ONE,t.ZERO,t.ONE,t.ZERO),t.blendColor(0,0,0,0),t.colorMask(!0,!0,!0,!0),t.clearColor(0,0,0,0),t.depthMask(!0),t.depthFunc(t.LESS),a.setReversed(!1),t.clearDepth(1),t.stencilMask(4294967295),t.stencilFunc(t.ALWAYS,0,4294967295),t.stencilOp(t.KEEP,t.KEEP,t.KEEP),t.clearStencil(0),t.cullFace(t.BACK),t.frontFace(t.CCW),t.polygonOffset(0,0),t.activeTexture(t.TEXTURE0),t.bindFramebuffer(t.FRAMEBUFFER,null),t.bindFramebuffer(t.DRAW_FRAMEBUFFER,null),t.bindFramebuffer(t.READ_FRAMEBUFFER,null),t.useProgram(null),t.lineWidth(1),t.scissor(0,0,t.canvas.width,t.canvas.height),t.viewport(0,0,t.canvas.width,t.canvas.height),t.pixelStorei(t.PACK_ALIGNMENT,4),t.pixelStorei(t.UNPACK_ALIGNMENT,4),t.pixelStorei(t.UNPACK_FLIP_Y_WEBGL,!1),t.pixelStorei(t.UNPACK_PREMULTIPLY_ALPHA_WEBGL,!1),t.pixelStorei(t.UNPACK_COLORSPACE_CONVERSION_WEBGL,t.BROWSER_DEFAULT_WEBGL),t.pixelStorei(t.PACK_ROW_LENGTH,0),t.pixelStorei(t.PACK_SKIP_PIXELS,0),t.pixelStorei(t.PACK_SKIP_ROWS,0),t.pixelStorei(t.UNPACK_ROW_LENGTH,0),t.pixelStorei(t.UNPACK_IMAGE_HEIGHT,0),t.pixelStorei(t.UNPACK_SKIP_PIXELS,0),t.pixelStorei(t.UNPACK_SKIP_ROWS,0),t.pixelStorei(t.UNPACK_SKIP_IMAGES,0),h={},p={},$=null,te={},u={},d=new WeakMap,v=[],M=null,m=!1,f=null,g=null,S=null,_=null,E=null,T=null,C=null,y=new Ye(0,0,0),b=0,R=!1,N=null,F=null,k=null,B=null,z=null,ut.set(0,0,t.canvas.width,t.canvas.height),qe.set(0,0,t.canvas.width,t.canvas.height),r.reset(),a.reset(),o.reset()}return{buffers:{color:r,depth:a,stencil:o},enable:ee,disable:ue,bindFramebuffer:Se,drawBuffers:ge,useProgram:Fe,setBlending:tt,setMaterial:_t,setFlipSided:Qe,setCullFace:Ut,setLineWidth:on,setPolygonOffset:Hn,setScissorTest:Pt,activeTexture:Yt,bindTexture:I,unbindTexture:Sn,compressedTexImage2D:ft,compressedTexImage3D:w,texImage2D:Q,texImage3D:j,pixelStorei:Te,getParameter:ae,updateUBOMapping:be,uniformBlockBinding:Re,texStorage2D:se,texStorage3D:re,texSubImage2D:x,texSubImage3D:P,compressedTexSubImage2D:H,compressedTexSubImage3D:Y,scissor:fe,viewport:oe,reset:Le}}function RU(t,e,n,i,s,r,a){let o=e.has("WEBGL_multisampled_render_to_texture")?e.get("WEBGL_multisampled_render_to_texture"):null,l=typeof navigator>"u"?!1:/OculusBrowser/g.test(navigator.userAgent),c=new Ne,h=new WeakMap,p=new Set,u,d=new WeakMap,v=!1;try{v=typeof OffscreenCanvas<"u"&&new OffscreenCanvas(1,1).getContext("2d")!==null}catch{}function M(w,x){return v?new OffscreenCanvas(w,x):hc("canvas")}function m(w,x,P){let H=1,Y=ft(w);if((Y.width>P||Y.height>P)&&(H=P/Math.max(Y.width,Y.height)),H<1)if(typeof HTMLImageElement<"u"&&w instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&w instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&w instanceof ImageBitmap||typeof VideoFrame<"u"&&w instanceof VideoFrame){let se=Math.floor(H*Y.width),re=Math.floor(H*Y.height);u===void 0&&(u=M(se,re));let Q=x?M(se,re):u;return Q.width=se,Q.height=re,Q.getContext("2d").drawImage(w,0,0,se,re),De("WebGLRenderer: Texture has been resized from ("+Y.width+"x"+Y.height+") to ("+se+"x"+re+")."),Q}else return"data"in w&&De("WebGLRenderer: Image in DataTexture is too big ("+Y.width+"x"+Y.height+")."),w;return w}function f(w){return w.generateMipmaps}function g(w){t.generateMipmap(w)}function S(w){return w.isWebGLCubeRenderTarget?t.TEXTURE_CUBE_MAP:w.isWebGL3DRenderTarget?t.TEXTURE_3D:w.isWebGLArrayRenderTarget||w.isCompressedArrayTexture?t.TEXTURE_2D_ARRAY:t.TEXTURE_2D}function _(w,x,P,H,Y,se=!1){if(w!==null){if(t[w]!==void 0)return t[w];De("WebGLRenderer: Attempt to use non-existing WebGL internal format '"+w+"'")}let re;H&&(re=e.get("EXT_texture_norm16"),re||De("WebGLRenderer: Unable to use normalized textures without EXT_texture_norm16 extension"));let Q=x;if(x===t.RED&&(P===t.FLOAT&&(Q=t.R32F),P===t.HALF_FLOAT&&(Q=t.R16F),P===t.UNSIGNED_BYTE&&(Q=t.R8),P===t.UNSIGNED_SHORT&&re&&(Q=re.R16_EXT),P===t.SHORT&&re&&(Q=re.R16_SNORM_EXT)),x===t.RED_INTEGER&&(P===t.UNSIGNED_BYTE&&(Q=t.R8UI),P===t.UNSIGNED_SHORT&&(Q=t.R16UI),P===t.UNSIGNED_INT&&(Q=t.R32UI),P===t.BYTE&&(Q=t.R8I),P===t.SHORT&&(Q=t.R16I),P===t.INT&&(Q=t.R32I)),x===t.RG&&(P===t.FLOAT&&(Q=t.RG32F),P===t.HALF_FLOAT&&(Q=t.RG16F),P===t.UNSIGNED_BYTE&&(Q=t.RG8),P===t.UNSIGNED_SHORT&&re&&(Q=re.RG16_EXT),P===t.SHORT&&re&&(Q=re.RG16_SNORM_EXT)),x===t.RG_INTEGER&&(P===t.UNSIGNED_BYTE&&(Q=t.RG8UI),P===t.UNSIGNED_SHORT&&(Q=t.RG16UI),P===t.UNSIGNED_INT&&(Q=t.RG32UI),P===t.BYTE&&(Q=t.RG8I),P===t.SHORT&&(Q=t.RG16I),P===t.INT&&(Q=t.RG32I)),x===t.RGB_INTEGER&&(P===t.UNSIGNED_BYTE&&(Q=t.RGB8UI),P===t.UNSIGNED_SHORT&&(Q=t.RGB16UI),P===t.UNSIGNED_INT&&(Q=t.RGB32UI),P===t.BYTE&&(Q=t.RGB8I),P===t.SHORT&&(Q=t.RGB16I),P===t.INT&&(Q=t.RGB32I)),x===t.RGBA_INTEGER&&(P===t.UNSIGNED_BYTE&&(Q=t.RGBA8UI),P===t.UNSIGNED_SHORT&&(Q=t.RGBA16UI),P===t.UNSIGNED_INT&&(Q=t.RGBA32UI),P===t.BYTE&&(Q=t.RGBA8I),P===t.SHORT&&(Q=t.RGBA16I),P===t.INT&&(Q=t.RGBA32I)),x===t.RGB&&(P===t.UNSIGNED_SHORT&&re&&(Q=re.RGB16_EXT),P===t.SHORT&&re&&(Q=re.RGB16_SNORM_EXT),P===t.UNSIGNED_INT_5_9_9_9_REV&&(Q=t.RGB9_E5),P===t.UNSIGNED_INT_10F_11F_11F_REV&&(Q=t.R11F_G11F_B10F)),x===t.RGBA){let j=se?uc:Ke.getTransfer(Y);P===t.FLOAT&&(Q=t.RGBA32F),P===t.HALF_FLOAT&&(Q=t.RGBA16F),P===t.UNSIGNED_BYTE&&(Q=j===ct?t.SRGB8_ALPHA8:t.RGBA8),P===t.UNSIGNED_SHORT&&re&&(Q=re.RGBA16_EXT),P===t.SHORT&&re&&(Q=re.RGBA16_SNORM_EXT),P===t.UNSIGNED_SHORT_4_4_4_4&&(Q=t.RGBA4),P===t.UNSIGNED_SHORT_5_5_5_1&&(Q=t.RGB5_A1)}return(Q===t.R16F||Q===t.R32F||Q===t.RG16F||Q===t.RG32F||Q===t.RGBA16F||Q===t.RGBA32F)&&e.get("EXT_color_buffer_float"),Q}function E(w,x){let P;return w?x===null||x===Wi||x===Or?P=t.DEPTH24_STENCIL8:x===fi?P=t.DEPTH32F_STENCIL8:x===Ko&&(P=t.DEPTH24_STENCIL8,De("DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.")):x===null||x===Wi||x===Or?P=t.DEPTH_COMPONENT24:x===fi?P=t.DEPTH_COMPONENT32F:x===Ko&&(P=t.DEPTH_COMPONENT16),P}function T(w,x){return f(w)===!0||w.isFramebufferTexture&&w.minFilter!==dn&&w.minFilter!==wt?Math.log2(Math.max(x.width,x.height))+1:w.mipmaps!==void 0&&w.mipmaps.length>0?w.mipmaps.length:w.isCompressedTexture&&Array.isArray(w.image)?x.mipmaps.length:1}function C(w){let x=w.target;x.removeEventListener("dispose",C),b(x),x.isVideoTexture&&h.delete(x),x.isHTMLTexture&&p.delete(x)}function y(w){let x=w.target;x.removeEventListener("dispose",y),N(x)}function b(w){let x=i.get(w);if(x.__webglInit===void 0)return;let P=w.source,H=d.get(P);if(H){let Y=H[x.__cacheKey];Y.usedTimes--,Y.usedTimes===0&&R(w),Object.keys(H).length===0&&d.delete(P)}i.remove(w)}function R(w){let x=i.get(w);t.deleteTexture(x.__webglTexture);let P=w.source,H=d.get(P);delete H[x.__cacheKey],a.memory.textures--}function N(w){let x=i.get(w);if(w.depthTexture&&(w.depthTexture.dispose(),i.remove(w.depthTexture)),w.isWebGLCubeRenderTarget)for(let H=0;H<6;H++){if(Array.isArray(x.__webglFramebuffer[H]))for(let Y=0;Y<x.__webglFramebuffer[H].length;Y++)t.deleteFramebuffer(x.__webglFramebuffer[H][Y]);else t.deleteFramebuffer(x.__webglFramebuffer[H]);x.__webglDepthbuffer&&t.deleteRenderbuffer(x.__webglDepthbuffer[H])}else{if(Array.isArray(x.__webglFramebuffer))for(let H=0;H<x.__webglFramebuffer.length;H++)t.deleteFramebuffer(x.__webglFramebuffer[H]);else t.deleteFramebuffer(x.__webglFramebuffer);if(x.__webglDepthbuffer&&t.deleteRenderbuffer(x.__webglDepthbuffer),x.__webglMultisampledFramebuffer&&t.deleteFramebuffer(x.__webglMultisampledFramebuffer),x.__webglColorRenderbuffer)for(let H=0;H<x.__webglColorRenderbuffer.length;H++)x.__webglColorRenderbuffer[H]&&t.deleteRenderbuffer(x.__webglColorRenderbuffer[H]);x.__webglDepthRenderbuffer&&t.deleteRenderbuffer(x.__webglDepthRenderbuffer)}let P=w.textures;for(let H=0,Y=P.length;H<Y;H++){let se=i.get(P[H]);se.__webglTexture&&(t.deleteTexture(se.__webglTexture),a.memory.textures--),i.remove(P[H])}i.remove(w)}let F=0;function k(){F=0}function B(){return F}function z(w){F=w}function Z(){let w=F;return w>=s.maxTextures&&De("WebGLTextures: Trying to use "+(w+1)+" texture units while this GPU supports only "+s.maxTextures),F+=1,w}function q(w){let x=[];return x.push(w.wrapS),x.push(w.wrapT),x.push(w.wrapR||0),x.push(w.magFilter),x.push(w.minFilter),x.push(w.anisotropy),x.push(w.internalFormat),x.push(w.format),x.push(w.type),x.push(w.generateMipmaps),x.push(w.premultiplyAlpha),x.push(w.flipY),x.push(w.unpackAlignment),x.push(w.colorSpace),x.join()}function ie(w,x){let P=i.get(w);if(w.isVideoTexture&&I(w),w.isRenderTargetTexture===!1&&w.isExternalTexture!==!0&&w.version>0&&P.__version!==w.version){let H=w.image;if(H===null)De("WebGLRenderer: Texture marked for update but no image data found.");else if(H.complete===!1)De("WebGLRenderer: Texture marked for update but image is incomplete");else{ue(P,w,x);return}}else w.isExternalTexture&&(P.__webglTexture=w.sourceTexture?w.sourceTexture:null);n.bindTexture(t.TEXTURE_2D,P.__webglTexture,t.TEXTURE0+x)}function W(w,x){let P=i.get(w);if(w.isRenderTargetTexture===!1&&w.version>0&&P.__version!==w.version){ue(P,w,x);return}else w.isExternalTexture&&(P.__webglTexture=w.sourceTexture?w.sourceTexture:null);n.bindTexture(t.TEXTURE_2D_ARRAY,P.__webglTexture,t.TEXTURE0+x)}function $(w,x){let P=i.get(w);if(w.isRenderTargetTexture===!1&&w.version>0&&P.__version!==w.version){ue(P,w,x);return}n.bindTexture(t.TEXTURE_3D,P.__webglTexture,t.TEXTURE0+x)}function te(w,x){let P=i.get(w);if(w.isCubeDepthTexture!==!0&&w.version>0&&P.__version!==w.version){Se(P,w,x);return}n.bindTexture(t.TEXTURE_CUBE_MAP,P.__webglTexture,t.TEXTURE0+x)}let we={[th]:t.REPEAT,[fs]:t.CLAMP_TO_EDGE,[nh]:t.MIRRORED_REPEAT},Me={[dn]:t.NEAREST,[T1]:t.NEAREST_MIPMAP_NEAREST,[bc]:t.NEAREST_MIPMAP_LINEAR,[wt]:t.LINEAR,[Rh]:t.LINEAR_MIPMAP_NEAREST,[Lr]:t.LINEAR_MIPMAP_LINEAR},ut={[C1]:t.NEVER,[B1]:t.ALWAYS,[R1]:t.LESS,[hd]:t.LEQUAL,[D1]:t.EQUAL,[dd]:t.GEQUAL,[U1]:t.GREATER,[I1]:t.NOTEQUAL};function qe(w,x){if(x.type===fi&&e.has("OES_texture_float_linear")===!1&&(x.magFilter===wt||x.magFilter===Rh||x.magFilter===bc||x.magFilter===Lr||x.minFilter===wt||x.minFilter===Rh||x.minFilter===bc||x.minFilter===Lr)&&De("WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device."),t.texParameteri(w,t.TEXTURE_WRAP_S,we[x.wrapS]),t.texParameteri(w,t.TEXTURE_WRAP_T,we[x.wrapT]),(w===t.TEXTURE_3D||w===t.TEXTURE_2D_ARRAY)&&t.texParameteri(w,t.TEXTURE_WRAP_R,we[x.wrapR]),t.texParameteri(w,t.TEXTURE_MAG_FILTER,Me[x.magFilter]),t.texParameteri(w,t.TEXTURE_MIN_FILTER,Me[x.minFilter]),x.compareFunction&&(t.texParameteri(w,t.TEXTURE_COMPARE_MODE,t.COMPARE_REF_TO_TEXTURE),t.texParameteri(w,t.TEXTURE_COMPARE_FUNC,ut[x.compareFunction])),e.has("EXT_texture_filter_anisotropic")===!0){if(x.magFilter===dn||x.minFilter!==bc&&x.minFilter!==Lr||x.type===fi&&e.has("OES_texture_float_linear")===!1)return;if(x.anisotropy>1||i.get(x).__currentAnisotropy){let P=e.get("EXT_texture_filter_anisotropic");t.texParameterf(w,P.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(x.anisotropy,s.getMaxAnisotropy())),i.get(x).__currentAnisotropy=x.anisotropy}}}function $e(w,x){let P=!1;w.__webglInit===void 0&&(w.__webglInit=!0,x.addEventListener("dispose",C));let H=x.source,Y=d.get(H);Y===void 0&&(Y={},d.set(H,Y));let se=q(x);if(se!==w.__cacheKey){Y[se]===void 0&&(Y[se]={texture:t.createTexture(),usedTimes:0},a.memory.textures++,P=!0),Y[se].usedTimes++;let re=Y[w.__cacheKey];re!==void 0&&(Y[w.__cacheKey].usedTimes--,re.usedTimes===0&&R(x)),w.__cacheKey=se,w.__webglTexture=Y[se].texture}return P}function X(w,x,P){return Math.floor(Math.floor(w/P)/x)}function ee(w,x,P,H){let se=w.updateRanges;if(se.length===0)n.texSubImage2D(t.TEXTURE_2D,0,0,0,x.width,x.height,P,H,x.data);else{se.sort((Te,fe)=>Te.start-fe.start);let re=0;for(let Te=1;Te<se.length;Te++){let fe=se[re],oe=se[Te],be=fe.start+fe.count,Re=X(oe.start,x.width,4),Le=X(fe.start,x.width,4);oe.start<=be+1&&Re===Le&&X(oe.start+oe.count-1,x.width,4)===Re?fe.count=Math.max(fe.count,oe.start+oe.count-fe.start):(++re,se[re]=oe)}se.length=re+1;let Q=n.getParameter(t.UNPACK_ROW_LENGTH),j=n.getParameter(t.UNPACK_SKIP_PIXELS),ae=n.getParameter(t.UNPACK_SKIP_ROWS);n.pixelStorei(t.UNPACK_ROW_LENGTH,x.width);for(let Te=0,fe=se.length;Te<fe;Te++){let oe=se[Te],be=Math.floor(oe.start/4),Re=Math.ceil(oe.count/4),Le=be%x.width,U=Math.floor(be/x.width),le=Re,K=1;n.pixelStorei(t.UNPACK_SKIP_PIXELS,Le),n.pixelStorei(t.UNPACK_SKIP_ROWS,U),n.texSubImage2D(t.TEXTURE_2D,0,Le,U,le,K,P,H,x.data)}w.clearUpdateRanges(),n.pixelStorei(t.UNPACK_ROW_LENGTH,Q),n.pixelStorei(t.UNPACK_SKIP_PIXELS,j),n.pixelStorei(t.UNPACK_SKIP_ROWS,ae)}}function ue(w,x,P){let H=t.TEXTURE_2D;(x.isDataArrayTexture||x.isCompressedArrayTexture)&&(H=t.TEXTURE_2D_ARRAY),x.isData3DTexture&&(H=t.TEXTURE_3D);let Y=$e(w,x),se=x.source;n.bindTexture(H,w.__webglTexture,t.TEXTURE0+P);let re=i.get(se);if(se.version!==re.__version||Y===!0){if(n.activeTexture(t.TEXTURE0+P),(typeof ImageBitmap<"u"&&x.image instanceof ImageBitmap)===!1){let K=Ke.getPrimaries(Ke.workingColorSpace),ce=x.colorSpace===hi?null:Ke.getPrimaries(x.colorSpace),me=x.colorSpace===hi||K===ce?t.NONE:t.BROWSER_DEFAULT_WEBGL;n.pixelStorei(t.UNPACK_FLIP_Y_WEBGL,x.flipY),n.pixelStorei(t.UNPACK_PREMULTIPLY_ALPHA_WEBGL,x.premultiplyAlpha),n.pixelStorei(t.UNPACK_COLORSPACE_CONVERSION_WEBGL,me)}n.pixelStorei(t.UNPACK_ALIGNMENT,x.unpackAlignment);let j=m(x.image,!1,s.maxTextureSize);j=Sn(x,j);let ae=r.convert(x.format,x.colorSpace),Te=r.convert(x.type),fe=_(x.internalFormat,ae,Te,x.normalized,x.colorSpace,x.isVideoTexture);qe(H,x);let oe,be=x.mipmaps,Re=x.isVideoTexture!==!0,Le=re.__version===void 0||Y===!0,U=se.dataReady,le=T(x,j);if(x.isDepthTexture)fe=E(x.format===ms,x.type),Le&&(Re?n.texStorage2D(t.TEXTURE_2D,1,fe,j.width,j.height):n.texImage2D(t.TEXTURE_2D,0,fe,j.width,j.height,0,ae,Te,null));else if(x.isDataTexture)if(be.length>0){Re&&Le&&n.texStorage2D(t.TEXTURE_2D,le,fe,be[0].width,be[0].height);for(let K=0,ce=be.length;K<ce;K++)oe=be[K],Re?U&&n.texSubImage2D(t.TEXTURE_2D,K,0,0,oe.width,oe.height,ae,Te,oe.data):n.texImage2D(t.TEXTURE_2D,K,fe,oe.width,oe.height,0,ae,Te,oe.data);x.generateMipmaps=!1}else Re?(Le&&n.texStorage2D(t.TEXTURE_2D,le,fe,j.width,j.height),U&&ee(x,j,ae,Te)):n.texImage2D(t.TEXTURE_2D,0,fe,j.width,j.height,0,ae,Te,j.data);else if(x.isCompressedTexture)if(x.isCompressedArrayTexture){Re&&Le&&n.texStorage3D(t.TEXTURE_2D_ARRAY,le,fe,be[0].width,be[0].height,j.depth);for(let K=0,ce=be.length;K<ce;K++)if(oe=be[K],x.format!==Ri)if(ae!==null)if(Re){if(U)if(x.layerUpdates.size>0){let me=U0(oe.width,oe.height,x.format,x.type);for(let ne of x.layerUpdates){let Ce=oe.data.subarray(ne*me/oe.data.BYTES_PER_ELEMENT,(ne+1)*me/oe.data.BYTES_PER_ELEMENT);n.compressedTexSubImage3D(t.TEXTURE_2D_ARRAY,K,0,0,ne,oe.width,oe.height,1,ae,Ce)}}else n.compressedTexSubImage3D(t.TEXTURE_2D_ARRAY,K,0,0,0,oe.width,oe.height,j.depth,ae,oe.data)}else n.compressedTexImage3D(t.TEXTURE_2D_ARRAY,K,fe,oe.width,oe.height,j.depth,0,oe.data,0,0);else De("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()");else Re?U&&n.texSubImage3D(t.TEXTURE_2D_ARRAY,K,0,0,0,oe.width,oe.height,j.depth,ae,Te,oe.data):n.texImage3D(t.TEXTURE_2D_ARRAY,K,fe,oe.width,oe.height,j.depth,0,ae,Te,oe.data);x.layerUpdates.size>0&&x.clearLayerUpdates()}else{Re&&Le&&n.texStorage2D(t.TEXTURE_2D,le,fe,be[0].width,be[0].height);for(let K=0,ce=be.length;K<ce;K++)oe=be[K],x.format!==Ri?ae!==null?Re?U&&n.compressedTexSubImage2D(t.TEXTURE_2D,K,0,0,oe.width,oe.height,ae,oe.data):n.compressedTexImage2D(t.TEXTURE_2D,K,fe,oe.width,oe.height,0,oe.data):De("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):Re?U&&n.texSubImage2D(t.TEXTURE_2D,K,0,0,oe.width,oe.height,ae,Te,oe.data):n.texImage2D(t.TEXTURE_2D,K,fe,oe.width,oe.height,0,ae,Te,oe.data)}else if(x.isDataArrayTexture)if(Re){if(Le&&n.texStorage3D(t.TEXTURE_2D_ARRAY,le,fe,j.width,j.height,j.depth),U)if(x.layerUpdates.size>0){let K=U0(j.width,j.height,x.format,x.type);for(let ce of x.layerUpdates){let me=j.data.subarray(ce*K/j.data.BYTES_PER_ELEMENT,(ce+1)*K/j.data.BYTES_PER_ELEMENT);n.texSubImage3D(t.TEXTURE_2D_ARRAY,0,0,0,ce,j.width,j.height,1,ae,Te,me)}x.clearLayerUpdates()}else n.texSubImage3D(t.TEXTURE_2D_ARRAY,0,0,0,0,j.width,j.height,j.depth,ae,Te,j.data)}else n.texImage3D(t.TEXTURE_2D_ARRAY,0,fe,j.width,j.height,j.depth,0,ae,Te,j.data);else if(x.isData3DTexture)Re?(Le&&n.texStorage3D(t.TEXTURE_3D,le,fe,j.width,j.height,j.depth),U&&n.texSubImage3D(t.TEXTURE_3D,0,0,0,0,j.width,j.height,j.depth,ae,Te,j.data)):n.texImage3D(t.TEXTURE_3D,0,fe,j.width,j.height,j.depth,0,ae,Te,j.data);else if(x.isFramebufferTexture){if(Le)if(Re)n.texStorage2D(t.TEXTURE_2D,le,fe,j.width,j.height);else{let K=j.width,ce=j.height;for(let me=0;me<le;me++)n.texImage2D(t.TEXTURE_2D,me,fe,K,ce,0,ae,Te,null),K>>=1,ce>>=1}}else if(x.isHTMLTexture){if("texElementImage2D"in t){let K=t.canvas;if(K.hasAttribute("layoutsubtree")||K.setAttribute("layoutsubtree","true"),j.parentNode!==K){K.appendChild(j),p.add(x),K.onpaint=ce=>{let me=ce.changedElements;for(let ne of p)me.includes(ne.image)&&(ne.needsUpdate=!0)},K.requestPaint();return}if(t.texElementImage2D.length===3)t.texElementImage2D(t.TEXTURE_2D,t.RGBA8,j);else{let me=t.RGBA,ne=t.RGBA,Ce=t.UNSIGNED_BYTE;t.texElementImage2D(t.TEXTURE_2D,0,me,ne,Ce,j)}t.texParameteri(t.TEXTURE_2D,t.TEXTURE_MIN_FILTER,t.LINEAR),t.texParameteri(t.TEXTURE_2D,t.TEXTURE_WRAP_S,t.CLAMP_TO_EDGE),t.texParameteri(t.TEXTURE_2D,t.TEXTURE_WRAP_T,t.CLAMP_TO_EDGE)}}else if(be.length>0){if(Re&&Le){let K=ft(be[0]);n.texStorage2D(t.TEXTURE_2D,le,fe,K.width,K.height)}for(let K=0,ce=be.length;K<ce;K++)oe=be[K],Re?U&&n.texSubImage2D(t.TEXTURE_2D,K,0,0,ae,Te,oe):n.texImage2D(t.TEXTURE_2D,K,fe,ae,Te,oe);x.generateMipmaps=!1}else if(Re){if(Le){let K=ft(j);n.texStorage2D(t.TEXTURE_2D,le,fe,K.width,K.height)}U&&n.texSubImage2D(t.TEXTURE_2D,0,0,0,ae,Te,j)}else n.texImage2D(t.TEXTURE_2D,0,fe,ae,Te,j);f(x)&&g(H),re.__version=se.version,x.onUpdate&&x.onUpdate(x)}w.__version=x.version}function Se(w,x,P){if(x.image.length!==6)return;let H=$e(w,x),Y=x.source;n.bindTexture(t.TEXTURE_CUBE_MAP,w.__webglTexture,t.TEXTURE0+P);let se=i.get(Y);if(Y.version!==se.__version||H===!0){n.activeTexture(t.TEXTURE0+P);let re=Ke.getPrimaries(Ke.workingColorSpace),Q=x.colorSpace===hi?null:Ke.getPrimaries(x.colorSpace),j=x.colorSpace===hi||re===Q?t.NONE:t.BROWSER_DEFAULT_WEBGL;n.pixelStorei(t.UNPACK_FLIP_Y_WEBGL,x.flipY),n.pixelStorei(t.UNPACK_PREMULTIPLY_ALPHA_WEBGL,x.premultiplyAlpha),n.pixelStorei(t.UNPACK_ALIGNMENT,x.unpackAlignment),n.pixelStorei(t.UNPACK_COLORSPACE_CONVERSION_WEBGL,j);let ae=x.isCompressedTexture||x.image[0].isCompressedTexture,Te=x.image[0]&&x.image[0].isDataTexture,fe=[];for(let ne=0;ne<6;ne++)!ae&&!Te?fe[ne]=m(x.image[ne],!0,s.maxCubemapSize):fe[ne]=Te?x.image[ne].image:x.image[ne],fe[ne]=Sn(x,fe[ne]);let oe=fe[0],be=r.convert(x.format,x.colorSpace),Re=r.convert(x.type),Le=_(x.internalFormat,be,Re,x.normalized,x.colorSpace),U=x.isVideoTexture!==!0,le=se.__version===void 0||H===!0,K=Y.dataReady,ce=T(x,oe);qe(t.TEXTURE_CUBE_MAP,x);let me;if(ae){U&&le&&n.texStorage2D(t.TEXTURE_CUBE_MAP,ce,Le,oe.width,oe.height);for(let ne=0;ne<6;ne++){me=fe[ne].mipmaps;for(let Ce=0;Ce<me.length;Ce++){let Ae=me[Ce];x.format!==Ri?be!==null?U?K&&n.compressedTexSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ne,Ce,0,0,Ae.width,Ae.height,be,Ae.data):n.compressedTexImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ne,Ce,Le,Ae.width,Ae.height,0,Ae.data):De("WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()"):U?K&&n.texSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ne,Ce,0,0,Ae.width,Ae.height,be,Re,Ae.data):n.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ne,Ce,Le,Ae.width,Ae.height,0,be,Re,Ae.data)}}}else{if(me=x.mipmaps,U&&le){me.length>0&&ce++;let ne=ft(fe[0]);n.texStorage2D(t.TEXTURE_CUBE_MAP,ce,Le,ne.width,ne.height)}for(let ne=0;ne<6;ne++)if(Te){U?K&&n.texSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ne,0,0,0,fe[ne].width,fe[ne].height,be,Re,fe[ne].data):n.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ne,0,Le,fe[ne].width,fe[ne].height,0,be,Re,fe[ne].data);for(let Ce=0;Ce<me.length;Ce++){let St=me[Ce].image[ne].image;U?K&&n.texSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ne,Ce+1,0,0,St.width,St.height,be,Re,St.data):n.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ne,Ce+1,Le,St.width,St.height,0,be,Re,St.data)}}else{U?K&&n.texSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ne,0,0,0,be,Re,fe[ne]):n.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ne,0,Le,be,Re,fe[ne]);for(let Ce=0;Ce<me.length;Ce++){let Ae=me[Ce];U?K&&n.texSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ne,Ce+1,0,0,be,Re,Ae.image[ne]):n.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ne,Ce+1,Le,be,Re,Ae.image[ne])}}}f(x)&&g(t.TEXTURE_CUBE_MAP),se.__version=Y.version,x.onUpdate&&x.onUpdate(x)}w.__version=x.version}function ge(w,x,P,H,Y,se){let re=r.convert(P.format,P.colorSpace),Q=r.convert(P.type),j=_(P.internalFormat,re,Q,P.normalized,P.colorSpace),ae=i.get(x),Te=i.get(P);if(Te.__renderTarget=x,!ae.__hasExternalTextures){let fe=Math.max(1,x.width>>se),oe=Math.max(1,x.height>>se);Y===t.TEXTURE_3D||Y===t.TEXTURE_2D_ARRAY?n.texImage3D(Y,se,j,fe,oe,x.depth,0,re,Q,null):n.texImage2D(Y,se,j,fe,oe,0,re,Q,null)}n.bindFramebuffer(t.FRAMEBUFFER,w),Yt(x)?o.framebufferTexture2DMultisampleEXT(t.FRAMEBUFFER,H,Y,Te.__webglTexture,0,Pt(x)):(Y===t.TEXTURE_2D||Y>=t.TEXTURE_CUBE_MAP_POSITIVE_X&&Y<=t.TEXTURE_CUBE_MAP_NEGATIVE_Z)&&t.framebufferTexture2D(t.FRAMEBUFFER,H,Y,Te.__webglTexture,se),n.bindFramebuffer(t.FRAMEBUFFER,null)}function Fe(w,x,P){if(t.bindRenderbuffer(t.RENDERBUFFER,w),x.depthBuffer){let H=x.depthTexture,Y=H&&H.isDepthTexture?H.type:null,se=E(x.stencilBuffer,Y),re=x.stencilBuffer?t.DEPTH_STENCIL_ATTACHMENT:t.DEPTH_ATTACHMENT;Yt(x)?o.renderbufferStorageMultisampleEXT(t.RENDERBUFFER,Pt(x),se,x.width,x.height):P?t.renderbufferStorageMultisample(t.RENDERBUFFER,Pt(x),se,x.width,x.height):t.renderbufferStorage(t.RENDERBUFFER,se,x.width,x.height),t.framebufferRenderbuffer(t.FRAMEBUFFER,re,t.RENDERBUFFER,w)}else{let H=x.textures;for(let Y=0;Y<H.length;Y++){let se=H[Y],re=r.convert(se.format,se.colorSpace),Q=r.convert(se.type),j=_(se.internalFormat,re,Q,se.normalized,se.colorSpace);Yt(x)?o.renderbufferStorageMultisampleEXT(t.RENDERBUFFER,Pt(x),j,x.width,x.height):P?t.renderbufferStorageMultisample(t.RENDERBUFFER,Pt(x),j,x.width,x.height):t.renderbufferStorage(t.RENDERBUFFER,j,x.width,x.height)}}t.bindRenderbuffer(t.RENDERBUFFER,null)}function en(w,x,P){let H=x.isWebGLCubeRenderTarget===!0;if(n.bindFramebuffer(t.FRAMEBUFFER,w),!(x.depthTexture&&x.depthTexture.isDepthTexture))throw new Error("THREE.WebGLTextures: renderTarget.depthTexture must be an instance of THREE.DepthTexture.");let Y=i.get(x.depthTexture);if(Y.__renderTarget=x,(!Y.__webglTexture||x.depthTexture.image.width!==x.width||x.depthTexture.image.height!==x.height)&&(x.depthTexture.image.width=x.width,x.depthTexture.image.height=x.height,x.depthTexture.needsUpdate=!0),H){if(Y.__webglInit===void 0&&(Y.__webglInit=!0,x.depthTexture.addEventListener("dispose",C)),Y.__webglTexture===void 0){Y.__webglTexture=t.createTexture(),n.bindTexture(t.TEXTURE_CUBE_MAP,Y.__webglTexture),qe(t.TEXTURE_CUBE_MAP,x.depthTexture);let ae=r.convert(x.depthTexture.format),Te=r.convert(x.depthTexture.type),fe;x.depthTexture.format===hs?fe=t.DEPTH_COMPONENT24:x.depthTexture.format===ms&&(fe=t.DEPTH24_STENCIL8);for(let oe=0;oe<6;oe++)t.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+oe,0,fe,x.width,x.height,0,ae,Te,null)}}else ie(x.depthTexture,0);let se=Y.__webglTexture,re=Pt(x),Q=H?t.TEXTURE_CUBE_MAP_POSITIVE_X+P:t.TEXTURE_2D,j=x.depthTexture.format===ms?t.DEPTH_STENCIL_ATTACHMENT:t.DEPTH_ATTACHMENT;if(x.depthTexture.format===hs)Yt(x)?o.framebufferTexture2DMultisampleEXT(t.FRAMEBUFFER,j,Q,se,0,re):t.framebufferTexture2D(t.FRAMEBUFFER,j,Q,se,0);else if(x.depthTexture.format===ms)Yt(x)?o.framebufferTexture2DMultisampleEXT(t.FRAMEBUFFER,j,Q,se,0,re):t.framebufferTexture2D(t.FRAMEBUFFER,j,Q,se,0);else throw new Error("THREE.WebGLTextures: Unknown depthTexture format.")}function Ve(w){let x=i.get(w),P=w.isWebGLCubeRenderTarget===!0;if(x.__boundDepthTexture!==w.depthTexture){let H=w.depthTexture;if(x.__depthDisposeCallback&&x.__depthDisposeCallback(),H){let Y=()=>{delete x.__boundDepthTexture,delete x.__depthDisposeCallback,H.removeEventListener("dispose",Y)};H.addEventListener("dispose",Y),x.__depthDisposeCallback=Y}x.__boundDepthTexture=H}if(w.depthTexture&&!x.__autoAllocateDepthBuffer)if(P)for(let H=0;H<6;H++)en(x.__webglFramebuffer[H],w,H);else{let H=w.texture.mipmaps;H&&H.length>0?en(x.__webglFramebuffer[0],w,0):en(x.__webglFramebuffer,w,0)}else if(P){x.__webglDepthbuffer=[];for(let H=0;H<6;H++)if(n.bindFramebuffer(t.FRAMEBUFFER,x.__webglFramebuffer[H]),x.__webglDepthbuffer[H]===void 0)x.__webglDepthbuffer[H]=t.createRenderbuffer(),Fe(x.__webglDepthbuffer[H],w,!1);else{let Y=w.stencilBuffer?t.DEPTH_STENCIL_ATTACHMENT:t.DEPTH_ATTACHMENT,se=x.__webglDepthbuffer[H];t.bindRenderbuffer(t.RENDERBUFFER,se),t.framebufferRenderbuffer(t.FRAMEBUFFER,Y,t.RENDERBUFFER,se)}}else{let H=w.texture.mipmaps;if(H&&H.length>0?n.bindFramebuffer(t.FRAMEBUFFER,x.__webglFramebuffer[0]):n.bindFramebuffer(t.FRAMEBUFFER,x.__webglFramebuffer),x.__webglDepthbuffer===void 0)x.__webglDepthbuffer=t.createRenderbuffer(),Fe(x.__webglDepthbuffer,w,!1);else{let Y=w.stencilBuffer?t.DEPTH_STENCIL_ATTACHMENT:t.DEPTH_ATTACHMENT,se=x.__webglDepthbuffer;t.bindRenderbuffer(t.RENDERBUFFER,se),t.framebufferRenderbuffer(t.FRAMEBUFFER,Y,t.RENDERBUFFER,se)}}n.bindFramebuffer(t.FRAMEBUFFER,null)}function tt(w,x,P){let H=i.get(w);x!==void 0&&ge(H.__webglFramebuffer,w,w.texture,t.COLOR_ATTACHMENT0,t.TEXTURE_2D,0),P!==void 0&&Ve(w)}function _t(w){let x=w.texture,P=i.get(w),H=i.get(x);w.addEventListener("dispose",y);let Y=w.textures,se=w.isWebGLCubeRenderTarget===!0,re=Y.length>1;if(re||(H.__webglTexture===void 0&&(H.__webglTexture=t.createTexture()),H.__version=x.version,a.memory.textures++),se){P.__webglFramebuffer=[];for(let Q=0;Q<6;Q++)if(x.mipmaps&&x.mipmaps.length>0){P.__webglFramebuffer[Q]=[];for(let j=0;j<x.mipmaps.length;j++)P.__webglFramebuffer[Q][j]=t.createFramebuffer()}else P.__webglFramebuffer[Q]=t.createFramebuffer()}else{if(x.mipmaps&&x.mipmaps.length>0){P.__webglFramebuffer=[];for(let Q=0;Q<x.mipmaps.length;Q++)P.__webglFramebuffer[Q]=t.createFramebuffer()}else P.__webglFramebuffer=t.createFramebuffer();if(re)for(let Q=0,j=Y.length;Q<j;Q++){let ae=i.get(Y[Q]);ae.__webglTexture===void 0&&(ae.__webglTexture=t.createTexture(),a.memory.textures++)}if(w.samples>0&&Yt(w)===!1){P.__webglMultisampledFramebuffer=t.createFramebuffer(),P.__webglColorRenderbuffer=[],n.bindFramebuffer(t.FRAMEBUFFER,P.__webglMultisampledFramebuffer);for(let Q=0;Q<Y.length;Q++){let j=Y[Q];P.__webglColorRenderbuffer[Q]=t.createRenderbuffer(),t.bindRenderbuffer(t.RENDERBUFFER,P.__webglColorRenderbuffer[Q]);let ae=r.convert(j.format,j.colorSpace),Te=r.convert(j.type),fe=_(j.internalFormat,ae,Te,j.normalized,j.colorSpace,w.isXRRenderTarget===!0),oe=Pt(w);t.renderbufferStorageMultisample(t.RENDERBUFFER,oe,fe,w.width,w.height),t.framebufferRenderbuffer(t.FRAMEBUFFER,t.COLOR_ATTACHMENT0+Q,t.RENDERBUFFER,P.__webglColorRenderbuffer[Q])}t.bindRenderbuffer(t.RENDERBUFFER,null),w.depthBuffer&&(P.__webglDepthRenderbuffer=t.createRenderbuffer(),Fe(P.__webglDepthRenderbuffer,w,!0)),n.bindFramebuffer(t.FRAMEBUFFER,null)}}if(se){n.bindTexture(t.TEXTURE_CUBE_MAP,H.__webglTexture),qe(t.TEXTURE_CUBE_MAP,x);for(let Q=0;Q<6;Q++)if(x.mipmaps&&x.mipmaps.length>0)for(let j=0;j<x.mipmaps.length;j++)ge(P.__webglFramebuffer[Q][j],w,x,t.COLOR_ATTACHMENT0,t.TEXTURE_CUBE_MAP_POSITIVE_X+Q,j);else ge(P.__webglFramebuffer[Q],w,x,t.COLOR_ATTACHMENT0,t.TEXTURE_CUBE_MAP_POSITIVE_X+Q,0);f(x)&&g(t.TEXTURE_CUBE_MAP),n.unbindTexture()}else if(re){for(let Q=0,j=Y.length;Q<j;Q++){let ae=Y[Q],Te=i.get(ae),fe=t.TEXTURE_2D;(w.isWebGL3DRenderTarget||w.isWebGLArrayRenderTarget)&&(fe=w.isWebGL3DRenderTarget?t.TEXTURE_3D:t.TEXTURE_2D_ARRAY),n.bindTexture(fe,Te.__webglTexture),qe(fe,ae),ge(P.__webglFramebuffer,w,ae,t.COLOR_ATTACHMENT0+Q,fe,0),f(ae)&&g(fe)}n.unbindTexture()}else{let Q=t.TEXTURE_2D;if((w.isWebGL3DRenderTarget||w.isWebGLArrayRenderTarget)&&(Q=w.isWebGL3DRenderTarget?t.TEXTURE_3D:t.TEXTURE_2D_ARRAY),n.bindTexture(Q,H.__webglTexture),qe(Q,x),x.mipmaps&&x.mipmaps.length>0)for(let j=0;j<x.mipmaps.length;j++)ge(P.__webglFramebuffer[j],w,x,t.COLOR_ATTACHMENT0,Q,j);else ge(P.__webglFramebuffer,w,x,t.COLOR_ATTACHMENT0,Q,0);f(x)&&g(Q),n.unbindTexture()}w.depthBuffer&&Ve(w)}function Qe(w){let x=w.textures;for(let P=0,H=x.length;P<H;P++){let Y=x[P];if(f(Y)){let se=S(w),re=i.get(Y).__webglTexture;n.bindTexture(se,re),g(se),n.unbindTexture()}}}let Ut=[],on=[];function Hn(w){if(w.samples>0){if(Yt(w)===!1){let x=w.textures,P=w.width,H=w.height,Y=t.COLOR_BUFFER_BIT,se=w.stencilBuffer?t.DEPTH_STENCIL_ATTACHMENT:t.DEPTH_ATTACHMENT,re=i.get(w),Q=x.length>1;if(Q)for(let ae=0;ae<x.length;ae++)n.bindFramebuffer(t.FRAMEBUFFER,re.__webglMultisampledFramebuffer),t.framebufferRenderbuffer(t.FRAMEBUFFER,t.COLOR_ATTACHMENT0+ae,t.RENDERBUFFER,null),n.bindFramebuffer(t.FRAMEBUFFER,re.__webglFramebuffer),t.framebufferTexture2D(t.DRAW_FRAMEBUFFER,t.COLOR_ATTACHMENT0+ae,t.TEXTURE_2D,null,0);n.bindFramebuffer(t.READ_FRAMEBUFFER,re.__webglMultisampledFramebuffer);let j=w.texture.mipmaps;j&&j.length>0?n.bindFramebuffer(t.DRAW_FRAMEBUFFER,re.__webglFramebuffer[0]):n.bindFramebuffer(t.DRAW_FRAMEBUFFER,re.__webglFramebuffer);for(let ae=0;ae<x.length;ae++){if(w.resolveDepthBuffer&&(w.depthBuffer&&(Y|=t.DEPTH_BUFFER_BIT),w.stencilBuffer&&w.resolveStencilBuffer&&(Y|=t.STENCIL_BUFFER_BIT)),Q){t.framebufferRenderbuffer(t.READ_FRAMEBUFFER,t.COLOR_ATTACHMENT0,t.RENDERBUFFER,re.__webglColorRenderbuffer[ae]);let Te=i.get(x[ae]).__webglTexture;t.framebufferTexture2D(t.DRAW_FRAMEBUFFER,t.COLOR_ATTACHMENT0,t.TEXTURE_2D,Te,0)}t.blitFramebuffer(0,0,P,H,0,0,P,H,Y,t.NEAREST),l===!0&&(Ut.length=0,on.length=0,Ut.push(t.COLOR_ATTACHMENT0+ae),w.depthBuffer&&w.storeMultisampledDepthBuffer===!1&&(Ut.push(se),on.push(se),t.invalidateFramebuffer(t.DRAW_FRAMEBUFFER,on)),t.invalidateFramebuffer(t.READ_FRAMEBUFFER,Ut))}if(n.bindFramebuffer(t.READ_FRAMEBUFFER,null),n.bindFramebuffer(t.DRAW_FRAMEBUFFER,null),Q)for(let ae=0;ae<x.length;ae++){n.bindFramebuffer(t.FRAMEBUFFER,re.__webglMultisampledFramebuffer),t.framebufferRenderbuffer(t.FRAMEBUFFER,t.COLOR_ATTACHMENT0+ae,t.RENDERBUFFER,re.__webglColorRenderbuffer[ae]);let Te=i.get(x[ae]).__webglTexture;n.bindFramebuffer(t.FRAMEBUFFER,re.__webglFramebuffer),t.framebufferTexture2D(t.DRAW_FRAMEBUFFER,t.COLOR_ATTACHMENT0+ae,t.TEXTURE_2D,Te,0)}n.bindFramebuffer(t.DRAW_FRAMEBUFFER,re.__webglMultisampledFramebuffer)}else if(w.depthBuffer&&w.storeMultisampledDepthBuffer===!1&&l){let x=w.stencilBuffer?t.DEPTH_STENCIL_ATTACHMENT:t.DEPTH_ATTACHMENT;t.invalidateFramebuffer(t.DRAW_FRAMEBUFFER,[x])}}}function Pt(w){return Math.min(s.maxSamples,w.samples)}function Yt(w){let x=i.get(w);return w.samples>0&&e.has("WEBGL_multisampled_render_to_texture")===!0&&x.__useRenderToTexture!==!1}function I(w){let x=a.render.frame;h.get(w)!==x&&(h.set(w,x),w.update())}function Sn(w,x){let P=w.colorSpace,H=w.format,Y=w.type;return w.isCompressedTexture===!0||w.isVideoTexture===!0||P!==Gs&&P!==hi&&(Ke.getTransfer(P)===ct?(H!==Ri||Y!==Xt)&&De("WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType."):Ie("WebGLTextures: Unsupported texture color space:",P)),x}function ft(w){return typeof HTMLImageElement<"u"&&w instanceof HTMLImageElement?(c.width=w.naturalWidth||w.width,c.height=w.naturalHeight||w.height):typeof VideoFrame<"u"&&w instanceof VideoFrame?(c.width=w.displayWidth,c.height=w.displayHeight):(c.width=w.width,c.height=w.height),c}this.allocateTextureUnit=Z,this.resetTextureUnits=k,this.getTextureUnits=B,this.setTextureUnits=z,this.setTexture2D=ie,this.setTexture2DArray=W,this.setTexture3D=$,this.setTextureCube=te,this.rebindTextures=tt,this.setupRenderTarget=_t,this.updateRenderTargetMipmap=Qe,this.updateMultisampleRenderTarget=Hn,this.setupDepthRenderbuffer=Ve,this.setupFrameBufferTexture=ge,this.useMultisampledRTT=Yt,this.isReversedDepthBuffer=function(){return n.buffers.depth.getReversed()}}function DU(t,e){function n(i,s=hi){let r,a=Ke.getTransfer(s);if(i===Xt)return t.UNSIGNED_BYTE;if(i===Uh)return t.UNSIGNED_SHORT_4_4_4_4;if(i===Ih)return t.UNSIGNED_SHORT_5_5_5_1;if(i===S0)return t.UNSIGNED_INT_5_9_9_9_REV;if(i===A0)return t.UNSIGNED_INT_10F_11F_11F_REV;if(i===y0)return t.BYTE;if(i===_0)return t.SHORT;if(i===Ko)return t.UNSIGNED_SHORT;if(i===Dh)return t.INT;if(i===Wi)return t.UNSIGNED_INT;if(i===fi)return t.FLOAT;if(i===Xi)return t.HALF_FLOAT;if(i===M0)return t.ALPHA;if(i===E0)return t.RGB;if(i===Ri)return t.RGBA;if(i===hs)return t.DEPTH_COMPONENT;if(i===ms)return t.DEPTH_STENCIL;if(i===T0)return t.RED;if(i===Bh)return t.RED_INTEGER;if(i===Fr)return t.RG;if(i===Nh)return t.RG_INTEGER;if(i===Ph)return t.RGBA_INTEGER;if(i===wc||i===Cc||i===Rc||i===Dc)if(a===ct)if(r=e.get("WEBGL_compressed_texture_s3tc_srgb"),r!==null){if(i===wc)return r.COMPRESSED_SRGB_S3TC_DXT1_EXT;if(i===Cc)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;if(i===Rc)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;if(i===Dc)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT}else return null;else if(r=e.get("WEBGL_compressed_texture_s3tc"),r!==null){if(i===wc)return r.COMPRESSED_RGB_S3TC_DXT1_EXT;if(i===Cc)return r.COMPRESSED_RGBA_S3TC_DXT1_EXT;if(i===Rc)return r.COMPRESSED_RGBA_S3TC_DXT3_EXT;if(i===Dc)return r.COMPRESSED_RGBA_S3TC_DXT5_EXT}else return null;if(i===Lh||i===Oh||i===Fh||i===zh)if(r=e.get("WEBGL_compressed_texture_pvrtc"),r!==null){if(i===Lh)return r.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;if(i===Oh)return r.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;if(i===Fh)return r.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;if(i===zh)return r.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}else return null;if(i===Hh||i===Gh||i===Vh||i===kh||i===Wh||i===Uc||i===Xh)if(r=e.get("WEBGL_compressed_texture_etc"),r!==null){if(i===Hh||i===Gh)return a===ct?r.COMPRESSED_SRGB8_ETC2:r.COMPRESSED_RGB8_ETC2;if(i===Vh)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:r.COMPRESSED_RGBA8_ETC2_EAC;if(i===kh)return r.COMPRESSED_R11_EAC;if(i===Wh)return r.COMPRESSED_SIGNED_R11_EAC;if(i===Uc)return r.COMPRESSED_RG11_EAC;if(i===Xh)return r.COMPRESSED_SIGNED_RG11_EAC}else return null;if(i===Yh||i===qh||i===Qh||i===Zh||i===Kh||i===Jh||i===jh||i===$h||i===ed||i===td||i===nd||i===id||i===sd||i===rd)if(r=e.get("WEBGL_compressed_texture_astc"),r!==null){if(i===Yh)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:r.COMPRESSED_RGBA_ASTC_4x4_KHR;if(i===qh)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:r.COMPRESSED_RGBA_ASTC_5x4_KHR;if(i===Qh)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:r.COMPRESSED_RGBA_ASTC_5x5_KHR;if(i===Zh)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:r.COMPRESSED_RGBA_ASTC_6x5_KHR;if(i===Kh)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:r.COMPRESSED_RGBA_ASTC_6x6_KHR;if(i===Jh)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:r.COMPRESSED_RGBA_ASTC_8x5_KHR;if(i===jh)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:r.COMPRESSED_RGBA_ASTC_8x6_KHR;if(i===$h)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:r.COMPRESSED_RGBA_ASTC_8x8_KHR;if(i===ed)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:r.COMPRESSED_RGBA_ASTC_10x5_KHR;if(i===td)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:r.COMPRESSED_RGBA_ASTC_10x6_KHR;if(i===nd)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:r.COMPRESSED_RGBA_ASTC_10x8_KHR;if(i===id)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:r.COMPRESSED_RGBA_ASTC_10x10_KHR;if(i===sd)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:r.COMPRESSED_RGBA_ASTC_12x10_KHR;if(i===rd)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:r.COMPRESSED_RGBA_ASTC_12x12_KHR}else return null;if(i===ad||i===od||i===ld)if(r=e.get("EXT_texture_compression_bptc"),r!==null){if(i===ad)return a===ct?r.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:r.COMPRESSED_RGBA_BPTC_UNORM_EXT;if(i===od)return r.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;if(i===ld)return r.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT}else return null;if(i===cd||i===ud||i===Ic||i===fd)if(r=e.get("EXT_texture_compression_rgtc"),r!==null){if(i===cd)return r.COMPRESSED_RED_RGTC1_EXT;if(i===ud)return r.COMPRESSED_SIGNED_RED_RGTC1_EXT;if(i===Ic)return r.COMPRESSED_RED_GREEN_RGTC2_EXT;if(i===fd)return r.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT}else return null;return i===Or?t.UNSIGNED_INT_24_8:t[i]!==void 0?t[i]:null}return{convert:n}}var UU=`
void main() {

	gl_Position = vec4( position, 1.0 );

}`,IU=`
uniform sampler2DArray depthColor;
uniform float depthWidth;
uniform float depthHeight;

void main() {

	vec2 coord = vec2( gl_FragCoord.x / depthWidth, gl_FragCoord.y / depthHeight );

	if ( coord.x >= 1.0 ) {

		gl_FragDepth = texture( depthColor, vec3( coord.x - 1.0, coord.y, 1 ) ).r;

	} else {

		gl_FragDepth = texture( depthColor, vec3( coord.x, coord.y, 0 ) ).r;

	}

}`,K0=class{constructor(){this.texture=null,this.mesh=null,this.depthNear=0,this.depthFar=0}init(e,n){if(this.texture===null){let i=new _c(e.texture);(e.depthNear!==n.depthNear||e.depthFar!==n.depthFar)&&(this.depthNear=e.depthNear,this.depthFar=e.depthFar),this.texture=i}}getMesh(e){if(this.texture!==null&&this.mesh===null){let n=e.cameras[0].viewport,i=new Wt({vertexShader:UU,fragmentShader:IU,uniforms:{depthColor:{value:this.texture},depthWidth:{value:n.z},depthHeight:{value:n.w}}});this.mesh=new Rn(new ya(20,20),i)}return this.mesh}reset(){this.texture=null,this.mesh=null}getDepthTexture(){return this.texture}},J0=class extends Zn{constructor(e,n){super();let i=this,s=null,r=1,a=null,o="local-floor",l=1,c=null,h=null,p=null,u=null,d=null,v=null,M=typeof XRWebGLBinding<"u",m=new K0,f={},g=n.getContextAttributes(),S=null,_=null,E=[],T=[],C=new Ne,y=null,b=null,R=new Cn;R.viewport=new Ot;let N=new Cn;N.viewport=new Ot;let F=[R,N],k=new Th,B=null,z=null;this.cameraAutoUpdate=!0,this.enabled=!1,this.isPresenting=!1,this.getController=function(X){let ee=E[X];return ee===void 0&&(ee=new Xo,E[X]=ee),ee.getTargetRaySpace()},this.getControllerGrip=function(X){let ee=E[X];return ee===void 0&&(ee=new Xo,E[X]=ee),ee.getGripSpace()},this.getHand=function(X){let ee=E[X];return ee===void 0&&(ee=new Xo,E[X]=ee),ee.getHandSpace()};function Z(X){let ee=T.indexOf(X.inputSource);if(ee===-1)return;let ue=E[ee];ue!==void 0&&(ue.update(X.inputSource,X.frame,c||a),ue.dispatchEvent({type:X.type,data:X.inputSource}))}function q(){s.removeEventListener("select",Z),s.removeEventListener("selectstart",Z),s.removeEventListener("selectend",Z),s.removeEventListener("squeeze",Z),s.removeEventListener("squeezestart",Z),s.removeEventListener("squeezeend",Z),s.removeEventListener("end",q),s.removeEventListener("inputsourceschange",ie);for(let X=0;X<E.length;X++){let ee=T[X];ee!==null&&(T[X]=null,E[X].disconnect(ee))}B=null,z=null,m.reset();for(let X in f)delete f[X];if(e.setRenderTarget(S),d=null,u=null,p=null,s=null,_=null,$e.stop(),i.isPresenting=!1,e.setPixelRatio(y),e.setSize(C.width,C.height,!1),b!==null){let X=b.camera;X.fov=b.fov,X.zoom=b.zoom,X.updateProjectionMatrix(),b=null}i.dispatchEvent({type:"sessionend"})}this.setFramebufferScaleFactor=function(X){r=X,i.isPresenting===!0&&De("WebXRManager: Cannot change framebuffer scale while presenting.")},this.setReferenceSpaceType=function(X){o=X,i.isPresenting===!0&&De("WebXRManager: Cannot change reference space type while presenting.")},this.getReferenceSpace=function(){return c||a},this.setReferenceSpace=function(X){c=X},this.getBaseLayer=function(){return u!==null?u:d},this.getBinding=function(){return p===null&&M&&(p=new XRWebGLBinding(s,n)),p},this.getFrame=function(){return v},this.getSession=function(){return s},this.setSession=async function(X){if(s=X,s!==null){if(S=e.getRenderTarget(),s.addEventListener("select",Z),s.addEventListener("selectstart",Z),s.addEventListener("selectend",Z),s.addEventListener("squeeze",Z),s.addEventListener("squeezestart",Z),s.addEventListener("squeezeend",Z),s.addEventListener("end",q),s.addEventListener("inputsourceschange",ie),g.xrCompatible!==!0&&await n.makeXRCompatible(),y=e.getPixelRatio(),e.getSize(C),M&&"createProjectionLayer"in XRWebGLBinding.prototype){let ue=null,Se=null,ge=null;g.depth&&(ge=g.stencil?n.DEPTH24_STENCIL8:n.DEPTH_COMPONENT24,ue=g.stencil?ms:hs,Se=g.stencil?Or:Wi);let Fe={colorFormat:n.RGBA8,depthFormat:ge,scaleFactor:r};p=this.getBinding(),u=p.createProjectionLayer(Fe),s.updateRenderState({layers:[u]}),e.setPixelRatio(1),e.setSize(u.textureWidth,u.textureHeight,!1),_=new Ft(u.textureWidth,u.textureHeight,{format:Ri,type:Xt,depthTexture:new Ci(u.textureWidth,u.textureHeight,Se,void 0,void 0,void 0,void 0,void 0,void 0,ue),stencilBuffer:g.stencil,colorSpace:e.outputColorSpace,samples:g.antialias?4:0,resolveDepthBuffer:u.ignoreDepthValues===!1,resolveStencilBuffer:u.ignoreDepthValues===!1,storeMultisampledDepthBuffer:u.ignoreDepthValues===!1,storeMultisampledStencilBuffer:u.ignoreDepthValues===!1})}else{let ue={antialias:g.antialias,alpha:!0,depth:g.depth,stencil:g.stencil,framebufferScaleFactor:r};d=new XRWebGLLayer(s,n,ue),s.updateRenderState({baseLayer:d}),e.setPixelRatio(1),e.setSize(d.framebufferWidth,d.framebufferHeight,!1),_=new Ft(d.framebufferWidth,d.framebufferHeight,{format:Ri,type:Xt,colorSpace:e.outputColorSpace,stencilBuffer:g.stencil,resolveDepthBuffer:d.ignoreDepthValues===!1,resolveStencilBuffer:d.ignoreDepthValues===!1,storeMultisampledDepthBuffer:d.ignoreDepthValues===!1,storeMultisampledStencilBuffer:d.ignoreDepthValues===!1})}_.isXRRenderTarget=!0,this.setFoveation(l),c=null,a=await s.requestReferenceSpace(o),$e.setContext(s),$e.start(),i.isPresenting=!0,i.dispatchEvent({type:"sessionstart"})}},this.getEnvironmentBlendMode=function(){if(s!==null)return s.environmentBlendMode},this.getDepthTexture=function(){return m.getDepthTexture()};function ie(X){for(let ee=0;ee<X.removed.length;ee++){let ue=X.removed[ee],Se=T.indexOf(ue);Se>=0&&(T[Se]=null,E[Se].disconnect(ue))}for(let ee=0;ee<X.added.length;ee++){let ue=X.added[ee],Se=T.indexOf(ue);if(Se===-1){for(let Fe=0;Fe<E.length;Fe++)if(Fe>=T.length){T.push(ue),Se=Fe;break}else if(T[Fe]===null){T[Fe]=ue,Se=Fe;break}if(Se===-1)break}let ge=E[Se];ge&&ge.connect(ue)}}let W=new G,$=new G;function te(X,ee,ue){W.setFromMatrixPosition(ee.matrixWorld),$.setFromMatrixPosition(ue.matrixWorld);let Se=W.distanceTo($),ge=ee.projectionMatrix.elements,Fe=ue.projectionMatrix.elements,en=ge[14]/(ge[10]-1),Ve=ge[14]/(ge[10]+1),tt=(ge[9]+1)/ge[5],_t=(ge[9]-1)/ge[5],Qe=(ge[8]-1)/ge[0],Ut=(Fe[8]+1)/Fe[0],on=en*Qe,Hn=en*Ut,Pt=Se/(-Qe+Ut),Yt=Pt*-Qe;if(ee.matrixWorld.decompose(X.position,X.quaternion,X.scale),X.translateX(Yt),X.translateZ(Pt),X.matrixWorld.compose(X.position,X.quaternion,X.scale),X.matrixWorldInverse.copy(X.matrixWorld).invert(),ge[10]===-1)X.projectionMatrix.copy(ee.projectionMatrix),X.projectionMatrixInverse.copy(ee.projectionMatrixInverse);else{let I=en+Pt,Sn=Ve+Pt,ft=on-Yt,w=Hn+(Se-Yt),x=tt*Ve/Sn*I,P=_t*Ve/Sn*I;X.projectionMatrix.makePerspective(ft,w,x,P,I,Sn),X.projectionMatrixInverse.copy(X.projectionMatrix).invert()}}function we(X,ee){ee===null?X.matrixWorld.copy(X.matrix):X.matrixWorld.multiplyMatrices(ee.matrixWorld,X.matrix),X.matrixWorldInverse.copy(X.matrixWorld).invert()}this.updateCamera=function(X){if(s===null)return;let ee=X.near,ue=X.far;m.texture!==null&&(m.depthNear>0&&(ee=m.depthNear),m.depthFar>0&&(ue=m.depthFar)),k.near=N.near=R.near=ee,k.far=N.far=R.far=ue,(B!==k.near||z!==k.far)&&(s.updateRenderState({depthNear:k.near,depthFar:k.far}),B=k.near,z=k.far),k.layers.mask=X.layers.mask|6,R.layers.mask=k.layers.mask&-5,N.layers.mask=k.layers.mask&-3;let Se=X.parent,ge=k.cameras;we(k,Se);for(let Fe=0;Fe<ge.length;Fe++)we(ge[Fe],Se);ge.length===2?te(k,R,N):k.projectionMatrix.copy(R.projectionMatrix),b===null&&X.isPerspectiveCamera&&(b={camera:X,fov:X.fov,zoom:X.zoom}),Me(X,k,Se)};function Me(X,ee,ue){ue===null?X.matrix.copy(ee.matrixWorld):(X.matrix.copy(ue.matrixWorld),X.matrix.invert(),X.matrix.multiply(ee.matrixWorld)),X.matrix.decompose(X.position,X.quaternion,X.scale),X.updateMatrixWorld(!0),X.projectionMatrix.copy(ee.projectionMatrix),X.projectionMatrixInverse.copy(ee.projectionMatrixInverse),X.isPerspectiveCamera&&(X.fov=sh*2*Math.atan(1/X.projectionMatrix.elements[5]),X.zoom=1)}this.getCamera=function(){return k},this.getFoveation=function(){if(!(u===null&&d===null))return l},this.setFoveation=function(X){l=X,u!==null&&(u.fixedFoveation=X),d!==null&&d.fixedFoveation!==void 0&&(d.fixedFoveation=X)},this.hasDepthSensing=function(){return m.texture!==null},this.getDepthSensingMesh=function(){return m.getMesh(k)},this.getCameraTexture=function(X){return f[X]};let ut=null;function qe(X,ee){if(h=ee.getViewerPose(c||a),v=ee,h!==null){let ue=h.views;d!==null&&(e.setRenderTargetFramebuffer(_,d.framebuffer),e.setRenderTarget(_));let Se=!1;ue.length!==k.cameras.length&&(k.cameras.length=0,Se=!0);for(let Ve=0;Ve<ue.length;Ve++){let tt=ue[Ve],_t=null;if(d!==null)_t=d.getViewport(tt);else{let Ut=p.getViewSubImage(u,tt);_t=Ut.viewport,Ve===0&&(e.setRenderTargetTextures(_,Ut.colorTexture,Ut.depthStencilTexture),e.setRenderTarget(_))}let Qe=F[Ve];Qe===void 0&&(Qe=new Cn,Qe.layers.enable(Ve),Qe.viewport=new Ot,F[Ve]=Qe),Qe.matrix.fromArray(tt.transform.matrix),Qe.matrix.decompose(Qe.position,Qe.quaternion,Qe.scale),Qe.projectionMatrix.fromArray(tt.projectionMatrix),Qe.projectionMatrixInverse.copy(Qe.projectionMatrix).invert(),Qe.viewport.set(_t.x,_t.y,_t.width,_t.height),Ve===0&&(k.matrix.copy(Qe.matrix),k.matrix.decompose(k.position,k.quaternion,k.scale)),Se===!0&&k.cameras.push(Qe)}let ge=s.enabledFeatures;if(ge&&ge.includes("depth-sensing")&&s.depthUsage=="gpu-optimized"&&M){p=i.getBinding();let Ve=p.getDepthInformation(ue[0]);Ve&&Ve.isValid&&Ve.texture&&m.init(Ve,s.renderState)}if(ge&&ge.includes("camera-access")&&M){e.state.unbindTexture(),p=i.getBinding();for(let Ve=0;Ve<ue.length;Ve++){let tt=ue[Ve].camera;if(tt){let _t=f[tt];_t||(_t=new _c,f[tt]=_t);let Qe=p.getCameraImage(tt);_t.sourceTexture=Qe}}}}for(let ue=0;ue<E.length;ue++){let Se=T[ue],ge=E[ue];Se!==null&&ge!==void 0&&ge.update(Se,ee,c||a)}ut&&ut(X,ee),ee.detectedPlanes&&i.dispatchEvent({type:"planesdetected",data:ee}),v=null}let $e=new uM;$e.setAnimationLoop(qe),this.setAnimationLoop=function(X){ut=X},this.dispose=function(){}}},BU=new kt,gM=new Pe;gM.set(-1,0,0,0,1,0,0,0,1);function NU(t,e){function n(m,f){m.matrixAutoUpdate===!0&&m.updateMatrix(),f.value.copy(m.matrix)}function i(m,f){f.color.getRGB(m.fogColor.value,C0(t)),f.isFog?(m.fogNear.value=f.near,m.fogFar.value=f.far):f.isFogExp2&&(m.fogDensity.value=f.density)}function s(m,f,g,S,_){f.isNodeMaterial?f.uniformsNeedUpdate=!1:f.isMeshBasicMaterial?r(m,f):f.isMeshLambertMaterial?(r(m,f),f.envMap&&(m.envMapIntensity.value=f.envMapIntensity)):f.isMeshToonMaterial?(r(m,f),p(m,f)):f.isMeshPhongMaterial?(r(m,f),h(m,f),f.envMap&&(m.envMapIntensity.value=f.envMapIntensity)):f.isMeshStandardMaterial?(r(m,f),u(m,f),f.isMeshPhysicalMaterial&&d(m,f,_)):f.isMeshMatcapMaterial?(r(m,f),v(m,f)):f.isMeshDepthMaterial?r(m,f):f.isMeshDistanceMaterial?(r(m,f),M(m,f)):f.isMeshNormalMaterial?r(m,f):f.isLineBasicMaterial?(a(m,f),f.isLineDashedMaterial&&o(m,f)):f.isPointsMaterial?l(m,f,g,S):f.isSpriteMaterial?c(m,f):f.isShadowMaterial?(m.color.value.copy(f.color),m.opacity.value=f.opacity):f.isShaderMaterial&&(f.uniformsNeedUpdate=!1)}function r(m,f){m.opacity.value=f.opacity,f.color&&m.diffuse.value.copy(f.color),f.emissive&&m.emissive.value.copy(f.emissive).multiplyScalar(f.emissiveIntensity),f.map&&(m.map.value=f.map,n(f.map,m.mapTransform)),f.alphaMap&&(m.alphaMap.value=f.alphaMap,n(f.alphaMap,m.alphaMapTransform)),f.bumpMap&&(m.bumpMap.value=f.bumpMap,n(f.bumpMap,m.bumpMapTransform),m.bumpScale.value=f.bumpScale,f.side===$t&&(m.bumpScale.value*=-1)),f.normalMap&&(m.normalMap.value=f.normalMap,n(f.normalMap,m.normalMapTransform),m.normalScale.value.copy(f.normalScale),f.side===$t&&m.normalScale.value.negate()),f.displacementMap&&(m.displacementMap.value=f.displacementMap,n(f.displacementMap,m.displacementMapTransform),m.displacementScale.value=f.displacementScale,m.displacementBias.value=f.displacementBias),f.emissiveMap&&(m.emissiveMap.value=f.emissiveMap,n(f.emissiveMap,m.emissiveMapTransform)),f.specularMap&&(m.specularMap.value=f.specularMap,n(f.specularMap,m.specularMapTransform)),f.alphaTest>0&&(m.alphaTest.value=f.alphaTest);let g=e.get(f),S=g.envMap,_=g.envMapRotation;S&&(m.envMap.value=S,m.envMapRotation.value.setFromMatrix4(BU.makeRotationFromEuler(_)).transpose(),S.isCubeTexture&&S.isRenderTargetTexture===!1&&m.envMapRotation.value.premultiply(gM),m.reflectivity.value=f.reflectivity,m.ior.value=f.ior,m.refractionRatio.value=f.refractionRatio),f.lightMap&&(m.lightMap.value=f.lightMap,m.lightMapIntensity.value=f.lightMapIntensity,n(f.lightMap,m.lightMapTransform)),f.aoMap&&(m.aoMap.value=f.aoMap,m.aoMapIntensity.value=f.aoMapIntensity,n(f.aoMap,m.aoMapTransform))}function a(m,f){m.diffuse.value.copy(f.color),m.opacity.value=f.opacity,f.map&&(m.map.value=f.map,n(f.map,m.mapTransform))}function o(m,f){m.dashSize.value=f.dashSize,m.totalSize.value=f.dashSize+f.gapSize,m.scale.value=f.scale}function l(m,f,g,S){m.diffuse.value.copy(f.color),m.opacity.value=f.opacity,m.size.value=f.size*g,m.scale.value=S*.5,f.map&&(m.map.value=f.map,n(f.map,m.uvTransform)),f.alphaMap&&(m.alphaMap.value=f.alphaMap,n(f.alphaMap,m.alphaMapTransform)),f.alphaTest>0&&(m.alphaTest.value=f.alphaTest)}function c(m,f){m.diffuse.value.copy(f.color),m.opacity.value=f.opacity,m.rotation.value=f.rotation,f.map&&(m.map.value=f.map,n(f.map,m.mapTransform)),f.alphaMap&&(m.alphaMap.value=f.alphaMap,n(f.alphaMap,m.alphaMapTransform)),f.alphaTest>0&&(m.alphaTest.value=f.alphaTest)}function h(m,f){m.specular.value.copy(f.specular),m.shininess.value=Math.max(f.shininess,1e-4)}function p(m,f){f.gradientMap&&(m.gradientMap.value=f.gradientMap)}function u(m,f){m.metalness.value=f.metalness,f.metalnessMap&&(m.metalnessMap.value=f.metalnessMap,n(f.metalnessMap,m.metalnessMapTransform)),m.roughness.value=f.roughness,f.roughnessMap&&(m.roughnessMap.value=f.roughnessMap,n(f.roughnessMap,m.roughnessMapTransform)),f.envMap&&(m.envMapIntensity.value=f.envMapIntensity)}function d(m,f,g){m.ior.value=f.ior,f.sheen>0&&(m.sheenColor.value.copy(f.sheenColor).multiplyScalar(f.sheen),m.sheenRoughness.value=f.sheenRoughness,f.sheenColorMap&&(m.sheenColorMap.value=f.sheenColorMap,n(f.sheenColorMap,m.sheenColorMapTransform)),f.sheenRoughnessMap&&(m.sheenRoughnessMap.value=f.sheenRoughnessMap,n(f.sheenRoughnessMap,m.sheenRoughnessMapTransform))),f.clearcoat>0&&(m.clearcoat.value=f.clearcoat,m.clearcoatRoughness.value=f.clearcoatRoughness,f.clearcoatMap&&(m.clearcoatMap.value=f.clearcoatMap,n(f.clearcoatMap,m.clearcoatMapTransform)),f.clearcoatRoughnessMap&&(m.clearcoatRoughnessMap.value=f.clearcoatRoughnessMap,n(f.clearcoatRoughnessMap,m.clearcoatRoughnessMapTransform)),f.clearcoatNormalMap&&(m.clearcoatNormalMap.value=f.clearcoatNormalMap,n(f.clearcoatNormalMap,m.clearcoatNormalMapTransform),m.clearcoatNormalScale.value.copy(f.clearcoatNormalScale),f.side===$t&&m.clearcoatNormalScale.value.negate())),f.dispersion>0&&(m.dispersion.value=f.dispersion),f.retroreflectivity>0&&(m.retroreflectivity.value=f.retroreflectivity),f.iridescence>0&&(m.iridescence.value=f.iridescence,m.iridescenceIOR.value=f.iridescenceIOR,m.iridescenceThicknessMinimum.value=f.iridescenceThicknessRange[0],m.iridescenceThicknessMaximum.value=f.iridescenceThicknessRange[1],f.iridescenceMap&&(m.iridescenceMap.value=f.iridescenceMap,n(f.iridescenceMap,m.iridescenceMapTransform)),f.iridescenceThicknessMap&&(m.iridescenceThicknessMap.value=f.iridescenceThicknessMap,n(f.iridescenceThicknessMap,m.iridescenceThicknessMapTransform))),f.transmission>0&&(m.transmission.value=f.transmission,m.transmissionSamplerMap.value=g.texture,m.transmissionSamplerSize.value.set(g.width,g.height),f.transmissionMap&&(m.transmissionMap.value=f.transmissionMap,n(f.transmissionMap,m.transmissionMapTransform)),m.thickness.value=f.thickness,f.thicknessMap&&(m.thicknessMap.value=f.thicknessMap,n(f.thicknessMap,m.thicknessMapTransform)),m.attenuationDistance.value=f.attenuationDistance,m.attenuationColor.value.copy(f.attenuationColor)),f.anisotropy>0&&(m.anisotropyVector.value.set(f.anisotropy*Math.cos(f.anisotropyRotation),f.anisotropy*Math.sin(f.anisotropyRotation)),f.anisotropyMap&&(m.anisotropyMap.value=f.anisotropyMap,n(f.anisotropyMap,m.anisotropyMapTransform))),m.specularIntensity.value=f.specularIntensity,m.specularColor.value.copy(f.specularColor),f.specularColorMap&&(m.specularColorMap.value=f.specularColorMap,n(f.specularColorMap,m.specularColorMapTransform)),f.specularIntensityMap&&(m.specularIntensityMap.value=f.specularIntensityMap,n(f.specularIntensityMap,m.specularIntensityMapTransform))}function v(m,f){f.matcap&&(m.matcap.value=f.matcap)}function M(m,f){let g=e.get(f).light;m.referencePosition.value.setFromMatrixPosition(g.matrixWorld),m.nearDistance.value=g.shadow.camera.near,m.farDistance.value=g.shadow.camera.far}return{refreshFogUniforms:i,refreshMaterialUniforms:s}}function PU(t,e,n,i){let s={},r={},a=[],o=t.getParameter(t.MAX_UNIFORM_BUFFER_BINDINGS);function l(_,E){let T=E.program;i.uniformBlockBinding(_,T)}function c(_,E){let T=s[_.id];T===void 0&&(m(_),T=h(_),s[_.id]=T,_.addEventListener("dispose",g));let C=E.program;i.updateUBOMapping(_,C);let y=e.render.frame;r[_.id]!==y&&(u(_),r[_.id]=y)}function h(_){let E=p();_.__bindingPointIndex=E;let T=t.createBuffer(),C=_.__size,y=_.usage;return t.bindBuffer(t.UNIFORM_BUFFER,T),t.bufferData(t.UNIFORM_BUFFER,C,y),t.bindBuffer(t.UNIFORM_BUFFER,null),t.bindBufferBase(t.UNIFORM_BUFFER,E,T),T}function p(){for(let _=0;_<o;_++)if(a.indexOf(_)===-1)return a.push(_),_;return Ie("WebGLRenderer: Maximum number of simultaneously usable uniforms groups reached."),0}function u(_){let E=s[_.id],T=_.uniforms,C=_.__cache;t.bindBuffer(t.UNIFORM_BUFFER,E);for(let y=0,b=T.length;y<b;y++){let R=T[y];if(Array.isArray(R))for(let N=0,F=R.length;N<F;N++)d(R[N],y,N,C);else d(R,y,0,C)}t.bindBuffer(t.UNIFORM_BUFFER,null)}function d(_,E,T,C){if(M(_,E,T,C)===!0){let y=_.__offset,b=_.value;if(Array.isArray(b)){let R=0;for(let N=0;N<b.length;N++){let F=b[N],k=f(F);v(F,_.__data,R),typeof F!="number"&&typeof F!="boolean"&&!F.isMatrix3&&!ArrayBuffer.isView(F)&&(R+=k.storage/Float32Array.BYTES_PER_ELEMENT)}}else v(b,_.__data,0);t.bufferSubData(t.UNIFORM_BUFFER,y,_.__data)}}function v(_,E,T){typeof _=="number"||typeof _=="boolean"?E[0]=_:_.isMatrix3?(E[0]=_.elements[0],E[1]=_.elements[1],E[2]=_.elements[2],E[3]=0,E[4]=_.elements[3],E[5]=_.elements[4],E[6]=_.elements[5],E[7]=0,E[8]=_.elements[6],E[9]=_.elements[7],E[10]=_.elements[8],E[11]=0):ArrayBuffer.isView(_)?E.set(new _.constructor(_.buffer,_.byteOffset,E.length)):_.toArray(E,T)}function M(_,E,T,C){let y=_.value,b=E+"_"+T;if(C[b]===void 0)return typeof y=="number"||typeof y=="boolean"?C[b]=y:ArrayBuffer.isView(y)?C[b]=y.slice():C[b]=y.clone(),!0;{let R=C[b];if(typeof y=="number"||typeof y=="boolean"){if(R!==y)return C[b]=y,!0}else{if(ArrayBuffer.isView(y))return!0;if(R.equals(y)===!1)return R.copy(y),!0}}return!1}function m(_){let E=_.uniforms,T=0,C=16;for(let b=0,R=E.length;b<R;b++){let N=Array.isArray(E[b])?E[b]:[E[b]];for(let F=0,k=N.length;F<k;F++){let B=N[F],z=Array.isArray(B.value)?B.value:[B.value];for(let Z=0,q=z.length;Z<q;Z++){let ie=z[Z],W=f(ie),$=T%C,te=$%W.boundary,we=$+te;T+=te,we!==0&&C-we<W.storage&&(T+=C-we),B.__data=new Float32Array(W.storage/Float32Array.BYTES_PER_ELEMENT),B.__offset=T,T+=W.storage}}}let y=T%C;return y>0&&(T+=C-y),_.__size=T,_.__cache={},this}function f(_){let E={boundary:0,storage:0};return typeof _=="number"||typeof _=="boolean"?(E.boundary=4,E.storage=4):_.isVector2?(E.boundary=8,E.storage=8):_.isVector3||_.isColor?(E.boundary=16,E.storage=12):_.isVector4?(E.boundary=16,E.storage=16):_.isMatrix3?(E.boundary=48,E.storage=48):_.isMatrix4?(E.boundary=64,E.storage=64):_.isTexture?De("WebGLRenderer: Texture samplers can not be part of an uniforms group."):ArrayBuffer.isView(_)?(E.boundary=16,E.storage=_.byteLength):De("WebGLRenderer: Unsupported uniform value type.",_),E}function g(_){let E=_.target;E.removeEventListener("dispose",g);let T=a.indexOf(E.__bindingPointIndex);a.splice(T,1),t.deleteBuffer(s[E.id]),delete s[E.id],delete r[E.id]}function S(){for(let _ in s)t.deleteBuffer(s[_]);a=[],s={},r={}}return{bind:l,update:c,dispose:S}}var LU=new Uint16Array([12469,15057,12620,14925,13266,14620,13807,14376,14323,13990,14545,13625,14713,13328,14840,12882,14931,12528,14996,12233,15039,11829,15066,11525,15080,11295,15085,10976,15082,10705,15073,10495,13880,14564,13898,14542,13977,14430,14158,14124,14393,13732,14556,13410,14702,12996,14814,12596,14891,12291,14937,11834,14957,11489,14958,11194,14943,10803,14921,10506,14893,10278,14858,9960,14484,14039,14487,14025,14499,13941,14524,13740,14574,13468,14654,13106,14743,12678,14818,12344,14867,11893,14889,11509,14893,11180,14881,10751,14852,10428,14812,10128,14765,9754,14712,9466,14764,13480,14764,13475,14766,13440,14766,13347,14769,13070,14786,12713,14816,12387,14844,11957,14860,11549,14868,11215,14855,10751,14825,10403,14782,10044,14729,9651,14666,9352,14599,9029,14967,12835,14966,12831,14963,12804,14954,12723,14936,12564,14917,12347,14900,11958,14886,11569,14878,11247,14859,10765,14828,10401,14784,10011,14727,9600,14660,9289,14586,8893,14508,8533,15111,12234,15110,12234,15104,12216,15092,12156,15067,12010,15028,11776,14981,11500,14942,11205,14902,10752,14861,10393,14812,9991,14752,9570,14682,9252,14603,8808,14519,8445,14431,8145,15209,11449,15208,11451,15202,11451,15190,11438,15163,11384,15117,11274,15055,10979,14994,10648,14932,10343,14871,9936,14803,9532,14729,9218,14645,8742,14556,8381,14461,8020,14365,7603,15273,10603,15272,10607,15267,10619,15256,10631,15231,10614,15182,10535,15118,10389,15042,10167,14963,9787,14883,9447,14800,9115,14710,8665,14615,8318,14514,7911,14411,7507,14279,7198,15314,9675,15313,9683,15309,9712,15298,9759,15277,9797,15229,9773,15166,9668,15084,9487,14995,9274,14898,8910,14800,8539,14697,8234,14590,7790,14479,7409,14367,7067,14178,6621,15337,8619,15337,8631,15333,8677,15325,8769,15305,8871,15264,8940,15202,8909,15119,8775,15022,8565,14916,8328,14804,8009,14688,7614,14569,7287,14448,6888,14321,6483,14088,6171,15350,7402,15350,7419,15347,7480,15340,7613,15322,7804,15287,7973,15229,8057,15148,8012,15046,7846,14933,7611,14810,7357,14682,7069,14552,6656,14421,6316,14251,5948,14007,5528,15356,5942,15356,5977,15353,6119,15348,6294,15332,6551,15302,6824,15249,7044,15171,7122,15070,7050,14949,6861,14818,6611,14679,6349,14538,6067,14398,5651,14189,5311,13935,4958,15359,4123,15359,4153,15356,4296,15353,4646,15338,5160,15311,5508,15263,5829,15188,6042,15088,6094,14966,6001,14826,5796,14678,5543,14527,5287,14377,4985,14133,4586,13869,4257,15360,1563,15360,1642,15358,2076,15354,2636,15341,3350,15317,4019,15273,4429,15203,4732,15105,4911,14981,4932,14836,4818,14679,4621,14517,4386,14359,4156,14083,3795,13808,3437,15360,122,15360,137,15358,285,15355,636,15344,1274,15322,2177,15281,2765,15215,3223,15120,3451,14995,3569,14846,3567,14681,3466,14511,3305,14344,3121,14037,2800,13753,2467,15360,0,15360,1,15359,21,15355,89,15346,253,15325,479,15287,796,15225,1148,15133,1492,15008,1749,14856,1882,14685,1886,14506,1783,14324,1608,13996,1398,13702,1183]),vs=null;function OU(){return vs===null&&(vs=new ch(LU,16,16,Fr,Xi),vs.name="DFG_LUT",vs.minFilter=wt,vs.magFilter=wt,vs.wrapS=fs,vs.wrapT=fs,vs.generateMipmaps=!1,vs.needsUpdate=!0),vs}var xd=class{constructor(e={}){let{canvas:n=P1(),context:i=null,depth:s=!0,stencil:r=!1,alpha:a=!1,antialias:o=!1,premultipliedAlpha:l=!0,preserveDrawingBuffer:c=!1,powerPreference:h="default",failIfMajorPerformanceCaveat:p=!1,reversedDepthBuffer:u=!1,outputBufferType:d=Xt}=e;this.isWebGLRenderer=!0;let v;if(i!==null){if(typeof WebGLRenderingContext<"u"&&i instanceof WebGLRenderingContext)throw new Error("THREE.WebGLRenderer: WebGL 1 is not supported since r163.");v=i.getContextAttributes().alpha}else v=a;let M=d,m=new Set([Ph,Nh,Bh]),f=new Set([Xt,Wi,Ko,Or,Uh,Ih]),g=new Uint32Array(4),S=new Int32Array(4),_=new G,E=null,T=null,C=[],y=[],b=null;this.domElement=n,this.debug={checkShaderErrors:!0,diagnostics:{keywords:!1},onShaderError:null},this.autoClear=!0,this.autoClearColor=!0,this.autoClearDepth=!0,this.autoClearStencil=!0,this.sortObjects=!0,this.clippingPlanes=[],this.localClippingEnabled=!1,this.toneMapping=ki,this.toneMappingExposure=1,this.transmissionResolutionScale=1;let R=this,N=!1,F=null,k=null,B=null,z=null;this._outputColorSpace=Dt;let Z=0,q=0,ie=null,W=-1,$=null,te=new Ot,we=new Ot,Me=null,ut=new Ye(0),qe=0,$e=n.width,X=n.height,ee=1,ue=null,Se=null,ge=new Ot(0,0,$e,X),Fe=new Ot(0,0,$e,X),en=!1,Ve=new xc,tt=!1,_t=!1,Qe=new kt,Ut=new G,on=new Ot,Hn={background:null,fog:null,environment:null,overrideMaterial:null,isScene:!0},Pt=!1;function Yt(){return ie===null?ee:1}let I=i;function Sn(A,D){return n.getContext(A,D)}let ft,w,x,P,H,Y,se,re,Q,j,ae,Te,fe,oe,be,Re,Le,U,le,K,ce,me,ne;try{let A={alpha:!0,depth:s,stencil:r,antialias:o,premultipliedAlpha:l,preserveDrawingBuffer:c,powerPreference:h,failIfMajorPerformanceCaveat:p};if("setAttribute"in n&&n.setAttribute("data-engine",`three.js r${"186"}`),n.addEventListener("webglcontextlost",St,!1),n.addEventListener("webglcontextrestored",at,!1),n.addEventListener("webglcontextcreationerror",Di,!1),I===null){let D="webgl2";if(I=Sn(D,A),I===null)throw Sn(D)?new Error("THREE.WebGLRenderer: Error creating WebGL context with your selected attributes."):new Error("THREE.WebGLRenderer: Error creating WebGL context.")}Ce()}catch(A){throw n.removeEventListener("webglcontextlost",St,!1),n.removeEventListener("webglcontextrestored",at,!1),n.removeEventListener("webglcontextcreationerror",Di,!1),Ie("WebGLRenderer: "+A.message),A}function Ce(){ft=new WR(I),ft.init(),ce=new DU(I,ft),w=new NR(I,ft,e,ce),x=new CU(I,ft),w.reversedDepthBuffer&&u&&x.buffers.depth.setReversed(!0),k=I.createFramebuffer(),B=I.createFramebuffer(),z=I.createFramebuffer(),P=new qR(I),H=new pU,Y=new RU(I,ft,x,H,w,ce,P),se=new kR(R),re=new Zw(I),me=new IR(I,re),Q=new XR(I,re,P,me),j=new ZR(I,Q,re,me,P),U=new QR(I,w,Y),be=new PR(H),ae=new dU(R,se,ft,w,me,be),Te=new NU(R,H),fe=new gU,oe=new AU(ft),Le=new UR(R,se,x,j,v,l),Re=new wU(R,j,w),ne=new PU(I,P,w,x),le=new BR(I,ft,P),K=new YR(I,ft,P),P.programs=ae.programs,R.capabilities=w,R.extensions=ft,R.properties=H,R.renderLists=fe,R.shadowMap=Re,R.state=x,R.info=P}M!==Xt&&(b=new JR(M,n.width,n.height,o,s,r));let Ae=new J0(R,I);this.xr=Ae,this.getContext=function(){return I},this.getContextAttributes=function(){return I.getContextAttributes()},this.forceContextLoss=function(){let A=ft.get("WEBGL_lose_context");A&&A.loseContext()},this.forceContextRestore=function(){let A=ft.get("WEBGL_lose_context");A&&A.restoreContext()},this.getPixelRatio=function(){return ee},this.setPixelRatio=function(A){A!==void 0&&(ee=A,this.setSize($e,X,!1))},this.getSize=function(A){return A.set($e,X)},this.setSize=function(A,D,V=!0){if(Ae.isPresenting){De("WebGLRenderer: Can't change size while VR device is presenting.");return}$e=A,X=D,n.width=Math.floor(A*ee),n.height=Math.floor(D*ee),V===!0&&(n.style.width=A+"px",n.style.height=D+"px"),b!==null&&b.setSize(n.width,n.height),this.setViewport(0,0,A,D)},this.getDrawingBufferSize=function(A){return A.set($e*ee,X*ee).floor()},this.setDrawingBufferSize=function(A,D,V){$e=A,X=D,ee=V,n.width=Math.floor(A*V),n.height=Math.floor(D*V),this.setViewport(0,0,A,D)},this.setEffects=function(A){if(M===Xt){Ie("WebGLRenderer: setEffects() requires outputBufferType set to HalfFloatType or FloatType.");return}if(A){for(let D=0;D<A.length;D++)if(A[D].isOutputPass===!0){De("WebGLRenderer: OutputPass is not needed in setEffects(). Tone mapping and color space conversion are applied automatically.");break}}b.setEffects(A||[])},this.getCurrentViewport=function(A){return A.copy(te)},this.getViewport=function(A){return A.copy(ge)},this.setViewport=function(A,D,V,L){A.isVector4?ge.set(A.x,A.y,A.z,A.w):ge.set(A,D,V,L),x.viewport(te.copy(ge).multiplyScalar(ee).round())},this.getScissor=function(A){return A.copy(Fe)},this.setScissor=function(A,D,V,L){A.isVector4?Fe.set(A.x,A.y,A.z,A.w):Fe.set(A,D,V,L),x.scissor(we.copy(Fe).multiplyScalar(ee).round())},this.getScissorTest=function(){return en},this.setScissorTest=function(A){x.setScissorTest(en=A)},this.setOpaqueSort=function(A){ue=A},this.setTransparentSort=function(A){Se=A},this.getClearColor=function(A){return A.copy(Le.getClearColor())},this.setClearColor=function(){Le.setClearColor(...arguments)},this.getClearAlpha=function(){return Le.getClearAlpha()},this.setClearAlpha=function(){Le.setClearAlpha(...arguments)},this.clear=function(A=!0,D=!0,V=!0){let L=0;if(A){let O=!1;if(ie!==null){let pe=ie.texture.format;O=m.has(pe)}if(O){let pe=ie.texture.type,xe=f.has(pe),de=Le.getClearColor(),ye=Le.getClearAlpha(),Ee=de.r,ze=de.g,ke=de.b;xe?(g[0]=Ee,g[1]=ze,g[2]=ke,g[3]=ye,I.clearBufferuiv(I.COLOR,0,g)):(S[0]=Ee,S[1]=ze,S[2]=ke,S[3]=ye,I.clearBufferiv(I.COLOR,0,S))}else L|=I.COLOR_BUFFER_BIT}D&&(L|=I.DEPTH_BUFFER_BIT,this.state.buffers.depth.setMask(!0)),V&&(L|=I.STENCIL_BUFFER_BIT,this.state.buffers.stencil.setMask(4294967295)),L!==0&&I.clear(L)},this.clearColor=function(){this.clear(!0,!1,!1)},this.clearDepth=function(){this.clear(!1,!0,!1)},this.clearStencil=function(){this.clear(!1,!1,!0)},this.setNodesHandler=function(A){A.setRenderer(this),F=A},this.dispose=function(){n.removeEventListener("webglcontextlost",St,!1),n.removeEventListener("webglcontextrestored",at,!1),n.removeEventListener("webglcontextcreationerror",Di,!1),Le.dispose(),fe.dispose(),oe.dispose(),H.dispose(),se.dispose(),j.dispose(),me.dispose(),ne.dispose(),ae.dispose(),Ae.dispose(),Ae.removeEventListener("sessionstart",av),Ae.removeEventListener("sessionend",ov),Hr.stop()};function St(A){A.preventDefault(),w0("WebGLRenderer: Context Lost."),N=!0}function at(){w0("WebGLRenderer: Context Restored."),N=!1;let A=P.autoReset,D=Re.enabled,V=Re.autoUpdate,L=Re.needsUpdate,O=Re.type;Ce(),P.autoReset=A,Re.enabled=D,Re.autoUpdate=V,Re.needsUpdate=L,Re.type=O}function Di(A){Ie("WebGLRenderer: A WebGL context could not be created. Reason: ",A.statusMessage)}function Yi(A){let D=A.target;D.removeEventListener("dispose",Yi),UM(D)}function UM(A){IM(A),H.remove(A)}function IM(A){let D=H.get(A).programs;D!==void 0&&(D.forEach(function(V){ae.releaseProgram(V)}),A.isShaderMaterial&&ae.releaseShaderCache(A))}this.renderBufferDirect=function(A,D,V,L,O,pe){D===null&&(D=Hn);let xe=O.isMesh&&O.matrixWorld.determinantAffine()<0,de=PM(A,D,V,L,O);x.setMaterial(L,xe);let ye=V.index,Ee=1;if(L.wireframe===!0){if(ye=Q.getWireframeAttribute(V),ye===void 0)return;Ee=2}let ze=V.drawRange,ke=V.attributes.position,_e=ze.start*Ee,ot=(ze.start+ze.count)*Ee;pe!==null&&(_e=Math.max(_e,pe.start*Ee),ot=Math.min(ot,(pe.start+pe.count)*Ee)),ye!==null?(_e=Math.max(_e,0),ot=Math.min(ot,ye.count)):ke!=null&&(_e=Math.max(_e,0),ot=Math.min(ot,ke.count));let qt=ot-_e;if(qt<0||qt===1/0)return;me.setup(O,L,de,V,ye);let Ct,vt=le;if(ye!==null&&(Ct=re.get(ye),vt=K,vt.setIndex(Ct)),O.isMesh)L.wireframe===!0?(x.setLineWidth(L.wireframeLinewidth*Yt()),vt.setMode(I.LINES)):vt.setMode(I.TRIANGLES);else if(O.isLine){let An=L.linewidth;An===void 0&&(An=1),x.setLineWidth(An*Yt()),O.isLineSegments?vt.setMode(I.LINES):O.isLineLoop?vt.setMode(I.LINE_LOOP):vt.setMode(I.LINE_STRIP)}else O.isPoints?vt.setMode(I.POINTS):O.isSprite&&vt.setMode(I.TRIANGLES);if(O.isBatchedMesh)if(ft.get("WEBGL_multi_draw"))vt.renderMultiDraw(O._multiDrawStarts,O._multiDrawCounts,O._multiDrawCount);else{let An=O._multiDrawStarts,ve=O._multiDrawCounts,In=O._multiDrawCount,et=ye?re.get(ye).bytesPerElement:1,di=H.get(L).currentProgram.getUniforms();for(let qi=0;qi<In;qi++)di.setValue(I,"_gl_DrawID",qi),vt.render(An[qi]/et,ve[qi])}else if(O.isInstancedMesh)vt.renderInstances(_e,qt,O.count);else if(V.isInstancedBufferGeometry){let An=V._maxInstanceCount!==void 0?V._maxInstanceCount:1/0,ve=Math.min(V.instanceCount,An);vt.renderInstances(_e,qt,ve)}else vt.render(_e,qt)};function rv(A,D,V,L){F!==null&&A.isNodeMaterial&&F.setObject(L,A),tt===!0&&be.setState(A,V,!1),A.transparent===!0&&A.side===zn&&A.forceSinglePass===!1?(A.side=$t,A.needsUpdate=!0,Hc(A,D,L),A.side=ps,A.needsUpdate=!0,Hc(A,D,L),A.side=zn):Hc(A,D,L)}this.compile=function(A,D,V=null){V===null&&(V=A),F!==null&&F.renderStart(A,D,V),T=oe.get(V),T.init(D),y.push(T),V.traverseVisible(function(O){O.isLight&&O.layers.test(D.layers)&&(T.pushLight(O),O.castShadow&&T.pushShadow(O))}),A!==V&&A.traverseVisible(function(O){O.isLight&&O.layers.test(D.layers)&&(T.pushLight(O),O.castShadow&&T.pushShadow(O))}),T.setupLights(),F!==null&&F.updateLights(T.state.lightsArray),_t=this.localClippingEnabled,tt=be.init(this.clippingPlanes,_t),tt===!0&&be.setGlobalState(this.clippingPlanes,D),F!==null&&Re.render(T.state.shadowsArray,V,D);let L=new Set;return A.traverse(function(O){if(!(O.isMesh||O.isPoints||O.isLine||O.isSprite))return;let pe=O.material;if(pe)if(Array.isArray(pe))for(let xe=0;xe<pe.length;xe++){let de=pe[xe];rv(de,V,D,O),L.add(de)}else rv(pe,V,D,O),L.add(pe)}),T=y.pop(),F!==null&&F.renderEnd(),L},this.compileAsync=function(A,D,V=null){let L=this.compile(A,D,V);return new Promise(O=>{function pe(){if(L.forEach(function(xe){let ye=H.get(xe).currentProgram;(ye===void 0||ye.isReady())&&L.delete(xe)}),L.size===0){O(A);return}setTimeout(pe,10)}ft.get("KHR_parallel_shader_compile")!==null?pe():setTimeout(pe,10)})};let Td=null;function BM(A){Td&&Td(A)}function av(){Hr.stop()}function ov(){Hr.start()}let Hr=new uM;Hr.setAnimationLoop(BM),typeof self<"u"&&Hr.setContext(self),this.setAnimationLoop=function(A){Td=A,Ae.setAnimationLoop(A),A===null?Hr.stop():Hr.start()},Ae.addEventListener("sessionstart",av),Ae.addEventListener("sessionend",ov),this.render=function(A,D){if(D!==void 0&&D.isCamera!==!0){Ie("WebGLRenderer.render: camera is not an instance of THREE.Camera.");return}if(N===!0)return;F!==null&&F.renderStart(A,D);let V=Ae.enabled===!0&&Ae.isPresenting===!0,L=b!==null&&(ie===null||V)&&b.begin(R,ie);if(A.matrixWorldAutoUpdate===!0&&A.updateMatrixWorld(),D.parent===null&&D.matrixWorldAutoUpdate===!0&&D.updateMatrixWorld(),Ae.enabled===!0&&Ae.isPresenting===!0&&(b===null||b.isCompositing()===!1)&&(Ae.cameraAutoUpdate===!0&&Ae.updateCamera(D),D=Ae.getCamera()),A.isScene===!0&&A.onBeforeRender(R,A,D,ie),T=oe.get(A,y.length),T.init(D),T.state.textureUnits=Y.getTextureUnits(),y.push(T),Qe.multiplyMatrices(D.projectionMatrix,D.matrixWorldInverse),Ve.setFromProjectionMatrix(Qe,Gi,D.reversedDepth),_t=this.localClippingEnabled,tt=be.init(this.clippingPlanes,_t),E=fe.get(A,C.length),E.init(),C.push(E),Ae.enabled===!0&&Ae.isPresenting===!0){let xe=R.xr.getDepthSensingMesh();xe!==null&&bd(xe,D,-1/0,R.sortObjects)}bd(A,D,0,R.sortObjects),E.finish(),F!==null&&F.updateLights(T.state.lightsArray),R.sortObjects===!0&&E.sort(ue,Se),Pt=Ae.enabled===!1||Ae.isPresenting===!1||Ae.hasDepthSensing()===!1,Pt&&Le.addToRenderList(E,A),this.info.render.frame++,this.info.autoReset===!0&&this.info.reset(),tt===!0&&be.beginShadows();let O=T.state.shadowsArray;if(Re.render(O,A,D),tt===!0&&be.endShadows(),(L&&b.hasRenderPass())===!1){let xe=E.opaque,de=E.transmissive;if(T.setupLights(),D.isArrayCamera){let ye=D.cameras;if(de.length>0)for(let Ee=0,ze=ye.length;Ee<ze;Ee++){let ke=ye[Ee];cv(xe,de,A,ke)}Pt&&Le.render(A);for(let Ee=0,ze=ye.length;Ee<ze;Ee++){let ke=ye[Ee];lv(E,A,ke,ke.viewport)}}else de.length>0&&cv(xe,de,A,D),Pt&&Le.render(A),lv(E,A,D)}ie!==null&&q===0&&(Y.updateMultisampleRenderTarget(ie),Y.updateRenderTargetMipmap(ie)),L&&b.end(R),A.isScene===!0&&A.onAfterRender(R,A,D),me.resetDefaultState(),W=-1,$=null,y.pop(),y.length>0?(T=y[y.length-1],Y.setTextureUnits(T.state.textureUnits),tt===!0&&be.setGlobalState(R.clippingPlanes,T.state.camera)):T=null,C.pop(),C.length>0?E=C[C.length-1]:E=null,F!==null&&F.renderEnd()};function bd(A,D,V,L){if(A.visible===!1)return;if(A.layers.test(D.layers)){if(A.isGroup)V=A.renderOrder;else if(A.isLOD)A.autoUpdate===!0&&A.update(D);else if(A.isLightProbeGrid)T.pushLightProbeGrid(A);else if(A.isLight)T.pushLight(A),A.castShadow&&T.pushShadow(A);else if(A.isSprite){if(!A.frustumCulled||A.intersectsFrustum(Ve)){L&&on.setFromMatrixPosition(A.matrixWorld).applyMatrix4(Qe);let xe=j.update(A),de=A.material;de.visible&&E.push(A,xe,de,V,on.z,null,D)}}else if((A.isMesh||A.isLine||A.isPoints)&&(!A.frustumCulled||A.intersectsFrustum(Ve))){let xe=j.update(A),de=A.material;if(L&&(A.boundingSphere!==void 0?(A.boundingSphere===null&&A.computeBoundingSphere(),on.copy(A.boundingSphere.center)):(xe.boundingSphere===null&&xe.computeBoundingSphere(),on.copy(xe.boundingSphere.center)),on.applyMatrix4(A.matrixWorld).applyMatrix4(Qe)),Array.isArray(de)){let ye=xe.groups;for(let Ee=0,ze=ye.length;Ee<ze;Ee++){let ke=ye[Ee],_e=de[ke.materialIndex];_e&&_e.visible&&E.push(A,xe,_e,V,on.z,ke,D)}}else de.visible&&E.push(A,xe,de,V,on.z,null,D)}}let pe=A.children;for(let xe=0,de=pe.length;xe<de;xe++)bd(pe[xe],D,V,L)}function lv(A,D,V,L){let{opaque:O,transmissive:pe,transparent:xe}=A;T.setupLightsView(V),tt===!0&&be.setGlobalState(R.clippingPlanes,V),L&&x.viewport(te.copy(L)),O.length>0&&zc(O,D,V),pe.length>0&&zc(pe,D,V),xe.length>0&&zc(xe,D,V),x.buffers.depth.setTest(!0),x.buffers.depth.setMask(!0),x.buffers.color.setMask(!0),x.setPolygonOffset(!1)}function cv(A,D,V,L){if((V.isScene===!0?V.overrideMaterial:null)!==null)return;if(T.state.transmissionRenderTarget[L.id]===void 0){let _e=ft.has("EXT_color_buffer_half_float")||ft.has("EXT_color_buffer_float");T.state.transmissionRenderTarget[L.id]=new Ft(1,1,{generateMipmaps:!0,type:_e?Xi:Xt,minFilter:Lr,samples:Math.max(4,w.samples),stencilBuffer:r,resolveDepthBuffer:!1,resolveStencilBuffer:!1,storeMultisampledDepthBuffer:!1,storeMultisampledStencilBuffer:!1,colorSpace:Ke.workingColorSpace})}let pe=T.state.transmissionRenderTarget[L.id],xe=L.viewport||te;pe.setSize(xe.z*R.transmissionResolutionScale,xe.w*R.transmissionResolutionScale);let de=R.getRenderTarget(),ye=R.getActiveCubeFace(),Ee=R.getActiveMipmapLevel();R.setRenderTarget(pe),R.getClearColor(ut),qe=R.getClearAlpha(),qe<1&&R.setClearColor(16777215,.5),R.clear(),Pt&&Le.render(V);let ze=R.toneMapping;R.toneMapping=ki;let ke=L.viewport;if(L.viewport!==void 0&&(L.viewport=void 0),T.setupLightsView(L),tt===!0&&be.setGlobalState(R.clippingPlanes,L),zc(A,V,L),Y.updateMultisampleRenderTarget(pe),Y.updateRenderTargetMipmap(pe),ft.has("WEBGL_multisampled_render_to_texture")===!1){let _e=!1;for(let ot=0,qt=D.length;ot<qt;ot++){let Ct=D[ot],{object:vt,geometry:An,material:ve,group:In}=Ct;if(ve.side===zn&&vt.layers.test(L.layers)){let et=ve.side;ve.side=$t,ve.needsUpdate=!0,uv(vt,V,L,An,ve,In),ve.side=et,ve.needsUpdate=!0,_e=!0}}_e===!0&&(Y.updateMultisampleRenderTarget(pe),Y.updateRenderTargetMipmap(pe))}R.setRenderTarget(de,ye,Ee),R.setClearColor(ut,qe),ke!==void 0&&(L.viewport=ke),R.toneMapping=ze}function zc(A,D,V){let L=D.isScene===!0?D.overrideMaterial:null;for(let O=0,pe=A.length;O<pe;O++){let xe=A[O],{object:de,geometry:ye,group:Ee}=xe,ze=xe.material;ze.allowOverride===!0&&L!==null&&(ze=L),de.layers.test(V.layers)&&uv(de,D,V,ye,ze,Ee)}}function uv(A,D,V,L,O,pe){F!==null&&O.isNodeMaterial&&F.setObject(A,O),A.onBeforeRender(R,D,V,L,O,pe),A.modelViewMatrix.multiplyMatrices(V.matrixWorldInverse,A.matrixWorld),A.normalMatrix.getNormalMatrix(A.modelViewMatrix),O.onBeforeRender(R,D,V,L,A,pe),O.transparent===!0&&O.side===zn&&O.forceSinglePass===!1?(O.side=$t,O.needsUpdate=!0,R.renderBufferDirect(V,D,L,O,A,pe),O.side=ps,O.needsUpdate=!0,R.renderBufferDirect(V,D,L,O,A,pe),O.side=zn):R.renderBufferDirect(V,D,L,O,A,pe),A.onAfterRender(R,D,V,L,O,pe)}function Hc(A,D,V){D.isScene!==!0&&(D=Hn);let L=H.get(A),O=T.state.lights,pe=T.state.shadowsArray,xe=O.state.version,de=ae.getParameters(A,O.state,pe,D,V,T.state.lightProbeGridArray),ye=ae.getProgramCacheKey(de),Ee=L.programs;L.environment=A.isMeshStandardMaterial||A.isMeshLambertMaterial||A.isMeshPhongMaterial?D.environment:null,L.fog=D.fog;let ze=A.isMeshStandardMaterial||A.isMeshLambertMaterial&&!A.envMap||A.isMeshPhongMaterial&&!A.envMap;L.envMap=se.get(A.envMap||L.environment,ze),L.envMapRotation=L.environment!==null&&A.envMap===null?D.environmentRotation:A.envMapRotation,Ee===void 0&&(A.addEventListener("dispose",Yi),Ee=new Map,L.programs=Ee);let ke=Ee.get(ye);if(ke!==void 0){if(L.currentProgram===ke&&L.lightsStateVersion===xe)return hv(A,de),ke}else de.uniforms=ae.getUniforms(A),F!==null&&A.isNodeMaterial&&F.build(A,V,de),A.onBeforeCompile(de,R),ke=ae.acquireProgram(de,ye),Ee.set(ye,ke),L.uniforms=de.uniforms;let _e=L.uniforms;return(!A.isShaderMaterial&&!A.isRawShaderMaterial||A.clipping===!0)&&(_e.clippingPlanes=be.uniform),hv(A,de),L.needsLights=OM(A),L.lightsStateVersion=xe,L.needsLights&&(_e.ambientLightColor.value=O.state.ambient,_e.lightProbe.value=O.state.probe,_e.sunLights.value=O.state.sun,_e.sunLightShadows.value=O.state.sunShadow,_e.directionalLights.value=O.state.directional,_e.directionalLightShadows.value=O.state.directionalShadow,_e.spotLights.value=O.state.spot,_e.spotLightShadows.value=O.state.spotShadow,_e.rectAreaLights.value=O.state.rectArea,_e.ltc_1.value=O.state.rectAreaLTC1,_e.ltc_2.value=O.state.rectAreaLTC2,_e.pointLights.value=O.state.point,_e.pointLightShadows.value=O.state.pointShadow,_e.hemisphereLights.value=O.state.hemi,_e.sunShadowMatrix.value=O.state.sunShadowMatrix,_e.sunShadowCascade.value=O.state.sunShadowCascade,_e.directionalShadowMatrix.value=O.state.directionalShadowMatrix,_e.spotLightMatrix.value=O.state.spotLightMatrix,_e.spotLightMap.value=O.state.spotLightMap,_e.pointShadowMatrix.value=O.state.pointShadowMatrix),L.lightProbeGrid=T.state.lightProbeGridArray.length>0,L.currentProgram=ke,L.uniformsList=null,ke}function fv(A){if(A.uniformsList===null){let D=A.currentProgram.getUniforms();A.uniformsList=$o.seqWithValue(D.seq,A.uniforms)}return A.uniformsList}function hv(A,D){let V=H.get(A);V.outputColorSpace=D.outputColorSpace,V.batching=D.batching,V.batchingColor=D.batchingColor,V.instancing=D.instancing,V.instancingColor=D.instancingColor,V.instancingMorph=D.instancingMorph,V.skinning=D.skinning,V.morphTargets=D.morphTargets,V.morphNormals=D.morphNormals,V.morphColors=D.morphColors,V.morphTargetsCount=D.morphTargetsCount,V.numClippingPlanes=D.numClippingPlanes,V.numIntersection=D.numClipIntersection,V.vertexAlphas=D.vertexAlphas,V.vertexTangents=D.vertexTangents,V.toneMapping=D.toneMapping}function NM(A,D){if(A.length===0)return null;if(A.length===1)return A[0].texture!==null?A[0]:null;_.setFromMatrixPosition(D.matrixWorld);for(let V=0,L=A.length;V<L;V++){let O=A[V];if(O.texture!==null&&O.boundingBox.containsPoint(_))return O}return null}function PM(A,D,V,L,O){D.isScene!==!0&&(D=Hn),Y.resetTextureUnits();let pe=D.fog,xe=L.isMeshStandardMaterial||L.isMeshLambertMaterial||L.isMeshPhongMaterial?D.environment:null,de=ie===null?R.outputColorSpace:ie.isXRRenderTarget===!0?ie.texture.colorSpace:Ke.workingColorSpace,ye=L.isMeshStandardMaterial||L.isMeshLambertMaterial&&!L.envMap||L.isMeshPhongMaterial&&!L.envMap,Ee=se.get(L.envMap||xe,ye),ze=L.vertexColors===!0&&!!V.attributes.color&&V.attributes.color.itemSize===4,ke=!!V.attributes.tangent&&(!!L.normalMap||L.anisotropy>0),_e=!!V.morphAttributes.position,ot=!!V.morphAttributes.normal,qt=!!V.morphAttributes.color,Ct=ki;L.toneMapped&&(ie===null||ie.isXRRenderTarget===!0)&&(Ct=R.toneMapping);let vt=V.morphAttributes.position||V.morphAttributes.normal||V.morphAttributes.color,An=vt!==void 0?vt.length:0,ve=H.get(L),In=T.state.lights;if(tt===!0&&(_t===!0||A!==$)){let At=A===$&&L.id===W;be.setState(L,A,At)}let et=!1;L.version===ve.__version?(ve.needsLights&&ve.lightsStateVersion!==In.state.version||ve.outputColorSpace!==de||O.isBatchedMesh&&ve.batching===!1||!O.isBatchedMesh&&ve.batching===!0||O.isBatchedMesh&&ve.batchingColor===!0&&O._colorsTexture===null||O.isBatchedMesh&&ve.batchingColor===!1&&O._colorsTexture!==null||O.isInstancedMesh&&ve.instancing===!1||!O.isInstancedMesh&&ve.instancing===!0||O.isSkinnedMesh&&ve.skinning===!1||!O.isSkinnedMesh&&ve.skinning===!0||O.isInstancedMesh&&ve.instancingColor===!0&&O.instanceColor===null||O.isInstancedMesh&&ve.instancingColor===!1&&O.instanceColor!==null||O.isInstancedMesh&&ve.instancingMorph===!0&&O.morphTexture===null||O.isInstancedMesh&&ve.instancingMorph===!1&&O.morphTexture!==null||ve.envMap!==Ee||L.fog===!0&&ve.fog!==pe||ve.numClippingPlanes!==void 0&&(ve.numClippingPlanes!==be.numPlanes||ve.numIntersection!==be.numIntersection)||ve.vertexAlphas!==ze||ve.vertexTangents!==ke||ve.morphTargets!==_e||ve.morphNormals!==ot||ve.morphColors!==qt||ve.toneMapping!==Ct||ve.morphTargetsCount!==An||!!ve.lightProbeGrid!=T.state.lightProbeGridArray.length>0)&&(et=!0):(et=!0,ve.__version=L.version);let di=ve.currentProgram;et===!0&&(di=Hc(L,D,O),F&&L.isNodeMaterial&&F.onUpdateProgram(L,di,ve));let qi=!1,ks=!1,ba=!1,mt=di.getUniforms(),Gt=ve.uniforms;if(x.useProgram(di.program)&&(qi=!0,ks=!0,ba=!0),L.id!==W&&(W=L.id,ks=!0),ve.needsLights){let At=NM(T.state.lightProbeGridArray,O);ve.lightProbeGrid!==At&&(ve.lightProbeGrid=At,ks=!0)}if(qi||$!==A){x.buffers.depth.getReversed()&&A.reversedDepth!==!0&&(A._reversedDepth=!0,A.updateProjectionMatrix()),mt.setValue(I,"projectionMatrix",A.projectionMatrix),mt.setValue(I,"viewMatrix",A.matrixWorldInverse);let Xs=mt.map.cameraPosition;Xs!==void 0&&Xs.setValue(I,Ut.setFromMatrixPosition(A.matrixWorld)),w.logarithmicDepthBuffer&&mt.setValue(I,"logDepthBufFC",2/(Math.log(A.far+1)/Math.LN2)),(L.isMeshPhongMaterial||L.isMeshToonMaterial||L.isMeshLambertMaterial||L.isMeshBasicMaterial||L.isMeshStandardMaterial||L.isShaderMaterial)&&mt.setValue(I,"isOrthographic",A.isOrthographicCamera===!0),$!==A&&($=A,ks=!0,ba=!0)}if(ve.needsLights&&(In.state.sunShadowMap.length>0&&mt.setValue(I,"sunShadowMap",In.state.sunShadowMap,Y),In.state.directionalShadowMap.length>0&&mt.setValue(I,"directionalShadowMap",In.state.directionalShadowMap,Y),In.state.spotShadowMap.length>0&&mt.setValue(I,"spotShadowMap",In.state.spotShadowMap,Y),In.state.pointShadowMap.length>0&&mt.setValue(I,"pointShadowMap",In.state.pointShadowMap,Y)),O.isSkinnedMesh){mt.setOptional(I,O,"bindMatrix"),mt.setOptional(I,O,"bindMatrixInverse");let At=O.skeleton;At&&(At.boneTexture===null&&At.computeBoneTexture(),mt.setValue(I,"boneTexture",At.boneTexture,Y))}O.isBatchedMesh&&(mt.setOptional(I,O,"batchingTexture"),mt.setValue(I,"batchingTexture",O._matricesTexture,Y),mt.setOptional(I,O,"batchingIdTexture"),mt.setValue(I,"batchingIdTexture",O._indirectTexture,Y),mt.setOptional(I,O,"batchingColorTexture"),O._colorsTexture!==null&&mt.setValue(I,"batchingColorTexture",O._colorsTexture,Y));let Ws=V.morphAttributes;if((Ws.position!==void 0||Ws.normal!==void 0||Ws.color!==void 0)&&U.update(O,V,di),(ks||ve.receiveShadow!==O.receiveShadow)&&(ve.receiveShadow=O.receiveShadow,mt.setValue(I,"receiveShadow",O.receiveShadow)),(L.isMeshStandardMaterial||L.isMeshLambertMaterial||L.isMeshPhongMaterial)&&L.envMap===null&&D.environment!==null&&(Gt.envMapIntensity.value=D.environmentIntensity),Gt.dfgLUT!==void 0&&(Gt.dfgLUT.value=OU()),ks){if(mt.setValue(I,"toneMappingExposure",R.toneMappingExposure),ve.needsLights&&LM(Gt,ba),pe&&L.fog===!0&&Te.refreshFogUniforms(Gt,pe),Te.refreshMaterialUniforms(Gt,L,ee,X,T.state.transmissionRenderTarget[A.id]),ve.needsLights&&ve.lightProbeGrid){let At=ve.lightProbeGrid;Gt.probesSH.value=At.texture,Gt.probesMin.value.copy(At.boundingBox.min),Gt.probesMax.value.copy(At.boundingBox.max),Gt.probesResolution.value.copy(At.resolution)}$o.upload(I,fv(ve),Gt,Y)}if(L.isShaderMaterial&&L.uniformsNeedUpdate===!0&&($o.upload(I,fv(ve),Gt,Y),L.uniformsNeedUpdate=!1),L.isSpriteMaterial&&mt.setValue(I,"center",O.center),mt.setValue(I,"modelViewMatrix",O.modelViewMatrix),mt.setValue(I,"normalMatrix",O.normalMatrix),mt.setValue(I,"modelMatrix",O.matrixWorld),L.uniformsGroups!==void 0){let At=L.uniformsGroups;for(let Xs=0,wa=At.length;Xs<wa;Xs++){let pv=At[Xs];ne.update(pv,di),ne.bind(pv,di)}}return di}function LM(A,D){A.ambientLightColor.needsUpdate=D,A.lightProbe.needsUpdate=D,A.sunLights.needsUpdate=D,A.sunLightShadows.needsUpdate=D,A.directionalLights.needsUpdate=D,A.directionalLightShadows.needsUpdate=D,A.pointLights.needsUpdate=D,A.pointLightShadows.needsUpdate=D,A.spotLights.needsUpdate=D,A.spotLightShadows.needsUpdate=D,A.rectAreaLights.needsUpdate=D,A.hemisphereLights.needsUpdate=D}function OM(A){return A.isMeshLambertMaterial||A.isMeshToonMaterial||A.isMeshPhongMaterial||A.isMeshStandardMaterial||A.isShadowMaterial||A.isShaderMaterial&&A.lights===!0}this.getActiveCubeFace=function(){return Z},this.getActiveMipmapLevel=function(){return q},this.getRenderTarget=function(){return ie},this.setRenderTargetTextures=function(A,D,V){let L=H.get(A);L.__autoAllocateDepthBuffer=A.resolveDepthBuffer===!1,L.__autoAllocateDepthBuffer===!1&&(L.__useRenderToTexture=!1),H.get(A.texture).__webglTexture=D,H.get(A.depthTexture).__webglTexture=L.__autoAllocateDepthBuffer?void 0:V,L.__hasExternalTextures=!0},this.setRenderTargetFramebuffer=function(A,D){let V=H.get(A);V.__webglFramebuffer=D,V.__useDefaultFramebuffer=D===void 0},this.setRenderTarget=function(A,D=0,V=0){ie=A,Z=D,q=V;let L=null,O=!1,pe=!1;if(A){let de=H.get(A);if(de.__useDefaultFramebuffer!==void 0){x.bindFramebuffer(I.FRAMEBUFFER,de.__webglFramebuffer),te.copy(A.viewport),we.copy(A.scissor),Me=A.scissorTest,x.viewport(te),x.scissor(we),x.setScissorTest(Me),W=-1;return}else if(de.__webglFramebuffer===void 0)Y.setupRenderTarget(A);else if(de.__hasExternalTextures)Y.rebindTextures(A,H.get(A.texture).__webglTexture,H.get(A.depthTexture).__webglTexture);else if(A.depthBuffer){let ze=A.depthTexture;if(de.__boundDepthTexture!==ze){if(ze!==null&&H.has(ze)&&(A.width!==ze.image.width||A.height!==ze.image.height))throw new Error("THREE.WebGLRenderer: Attached DepthTexture is initialized to the incorrect size.");Y.setupDepthRenderbuffer(A)}}let ye=A.texture;(ye.isData3DTexture||ye.isDataArrayTexture||ye.isCompressedArrayTexture)&&(pe=!0);let Ee=H.get(A).__webglFramebuffer;A.isWebGLCubeRenderTarget?(Array.isArray(Ee[D])?L=Ee[D][V]:L=Ee[D],O=!0):A.samples>0&&Y.useMultisampledRTT(A)===!1?L=H.get(A).__webglMultisampledFramebuffer:Array.isArray(Ee)?L=Ee[V]:L=Ee,te.copy(A.viewport),we.copy(A.scissor),Me=A.scissorTest}else te.copy(ge).multiplyScalar(ee).floor(),we.copy(Fe).multiplyScalar(ee).floor(),Me=en;if(V!==0&&(L=k),x.bindFramebuffer(I.FRAMEBUFFER,L)&&x.drawBuffers(A,L),x.viewport(te),x.scissor(we),x.setScissorTest(Me),O){let de=H.get(A.texture);I.framebufferTexture2D(I.FRAMEBUFFER,I.COLOR_ATTACHMENT0,I.TEXTURE_CUBE_MAP_POSITIVE_X+D,de.__webglTexture,V)}else if(pe){let de=D;for(let ye=0;ye<A.textures.length;ye++){let Ee=H.get(A.textures[ye]);I.framebufferTextureLayer(I.FRAMEBUFFER,I.COLOR_ATTACHMENT0+ye,Ee.__webglTexture,V,de)}}else if(A!==null&&V!==0){let de=H.get(A.texture);I.framebufferTexture2D(I.FRAMEBUFFER,I.COLOR_ATTACHMENT0,I.TEXTURE_2D,de.__webglTexture,V)}W=-1};function dv(A){let D=H.get(A);return(D.__readFormat!==A.format||D.__readType!==A.type)&&(D.__readFormat=A.format,D.__readType=A.type,D.__formatReadable=w.textureFormatReadable(A.format),D.__typeReadable=w.textureTypeReadable(A.type)),D}this.readRenderTargetPixels=function(A,D,V,L,O,pe,xe,de=0){if(!(A&&A.isWebGLRenderTarget)){Ie("WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");return}let ye=H.get(A).__webglFramebuffer;if(A.isWebGLCubeRenderTarget&&xe!==void 0&&(ye=ye[xe]),ye){x.bindFramebuffer(I.FRAMEBUFFER,ye);try{let Ee=A.textures[de],ze=Ee.format,ke=Ee.type;A.textures.length>1&&I.readBuffer(I.COLOR_ATTACHMENT0+de);let _e=dv(Ee);if(_e.__formatReadable===!1){Ie("WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");return}if(_e.__typeReadable===!1){Ie("WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");return}D>=0&&D<=A.width-L&&V>=0&&V<=A.height-O&&I.readPixels(D,V,L,O,ce.convert(ze),ce.convert(ke),pe)}finally{let Ee=ie!==null?H.get(ie).__webglFramebuffer:null;x.bindFramebuffer(I.FRAMEBUFFER,Ee)}}},this.readRenderTargetPixelsAsync=async function(A,D,V,L,O,pe,xe,de=0){if(!(A&&A.isWebGLRenderTarget))throw new Error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");let ye=H.get(A).__webglFramebuffer;if(A.isWebGLCubeRenderTarget&&xe!==void 0&&(ye=ye[xe]),ye)if(D>=0&&D<=A.width-L&&V>=0&&V<=A.height-O){x.bindFramebuffer(I.FRAMEBUFFER,ye);let Ee=A.textures[de],ze=Ee.format,ke=Ee.type;A.textures.length>1&&I.readBuffer(I.COLOR_ATTACHMENT0+de);let _e=dv(Ee);if(_e.__formatReadable===!1)throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in RGBA or implementation defined format.");if(_e.__typeReadable===!1)throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in UnsignedByteType or implementation defined type.");let ot=I.createBuffer();I.bindBuffer(I.PIXEL_PACK_BUFFER,ot),I.bufferData(I.PIXEL_PACK_BUFFER,pe.byteLength,I.STREAM_READ),I.readPixels(D,V,L,O,ce.convert(ze),ce.convert(ke),0),I.bindBuffer(I.PIXEL_PACK_BUFFER,null);let qt=ie!==null?H.get(ie).__webglFramebuffer:null;x.bindFramebuffer(I.FRAMEBUFFER,qt);let Ct=I.fenceSync(I.SYNC_GPU_COMMANDS_COMPLETE,0);return I.flush(),await O1(I,Ct,4),I.bindBuffer(I.PIXEL_PACK_BUFFER,ot),I.getBufferSubData(I.PIXEL_PACK_BUFFER,0,pe),I.bindBuffer(I.PIXEL_PACK_BUFFER,null),I.deleteBuffer(ot),I.deleteSync(Ct),pe}else throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: requested read bounds are out of range.")},this.copyFramebufferToTexture=function(A,D=null,V=0){let L=Math.pow(2,-V),O=Math.floor(A.image.width*L),pe=Math.floor(A.image.height*L),xe=D!==null?D.x:0,de=D!==null?D.y:0;Y.setTexture2D(A,0),I.copyTexSubImage2D(I.TEXTURE_2D,V,0,0,xe,de,O,pe),x.unbindTexture()},this.copyTextureToTexture=function(A,D,V=null,L=null,O=0,pe=0){let xe,de,ye,Ee,ze,ke,_e,ot,qt,Ct=A.isCompressedTexture?A.mipmaps[pe]:A.image;if(V!==null)xe=V.max.x-V.min.x,de=V.max.y-V.min.y,ye=V.isBox3?V.max.z-V.min.z:1,Ee=V.min.x,ze=V.min.y,ke=V.isBox3?V.min.z:0;else{let Gt=Math.pow(2,-O);xe=Math.floor(Ct.width*Gt),de=Math.floor(Ct.height*Gt),A.isDataArrayTexture?ye=Ct.depth:A.isData3DTexture?ye=Math.floor(Ct.depth*Gt):ye=1,Ee=0,ze=0,ke=0}L!==null?(_e=L.x,ot=L.y,qt=L.z):(_e=0,ot=0,qt=0);let vt=ce.convert(D.format),An=ce.convert(D.type),ve;D.isData3DTexture?(Y.setTexture3D(D,0),ve=I.TEXTURE_3D):D.isDataArrayTexture||D.isCompressedArrayTexture?(Y.setTexture2DArray(D,0),ve=I.TEXTURE_2D_ARRAY):(Y.setTexture2D(D,0),ve=I.TEXTURE_2D),x.activeTexture(I.TEXTURE0),x.pixelStorei(I.UNPACK_FLIP_Y_WEBGL,D.flipY),x.pixelStorei(I.UNPACK_PREMULTIPLY_ALPHA_WEBGL,D.premultiplyAlpha),x.pixelStorei(I.UNPACK_ALIGNMENT,D.unpackAlignment);let In=x.getParameter(I.UNPACK_ROW_LENGTH),et=x.getParameter(I.UNPACK_IMAGE_HEIGHT),di=x.getParameter(I.UNPACK_SKIP_PIXELS),qi=x.getParameter(I.UNPACK_SKIP_ROWS),ks=x.getParameter(I.UNPACK_SKIP_IMAGES);x.pixelStorei(I.UNPACK_ROW_LENGTH,Ct.width),x.pixelStorei(I.UNPACK_IMAGE_HEIGHT,Ct.height),x.pixelStorei(I.UNPACK_SKIP_PIXELS,Ee),x.pixelStorei(I.UNPACK_SKIP_ROWS,ze),x.pixelStorei(I.UNPACK_SKIP_IMAGES,ke);let ba=A.isDataArrayTexture||A.isData3DTexture,mt=D.isDataArrayTexture||D.isData3DTexture;if(A.isDepthTexture){let Gt=H.get(A),Ws=H.get(D),At=H.get(Gt.__renderTarget),Xs=H.get(Ws.__renderTarget);x.bindFramebuffer(I.READ_FRAMEBUFFER,At.__webglFramebuffer),x.bindFramebuffer(I.DRAW_FRAMEBUFFER,Xs.__webglFramebuffer);for(let wa=0;wa<ye;wa++)ba&&(I.framebufferTextureLayer(I.READ_FRAMEBUFFER,I.COLOR_ATTACHMENT0,H.get(A).__webglTexture,O,ke+wa),I.framebufferTextureLayer(I.DRAW_FRAMEBUFFER,I.COLOR_ATTACHMENT0,H.get(D).__webglTexture,pe,qt+wa)),I.blitFramebuffer(Ee,ze,xe,de,_e,ot,xe,de,I.DEPTH_BUFFER_BIT,I.NEAREST);x.bindFramebuffer(I.READ_FRAMEBUFFER,null),x.bindFramebuffer(I.DRAW_FRAMEBUFFER,null)}else if(O!==0||A.isRenderTargetTexture||H.has(A)){let Gt=H.get(A),Ws=H.get(D);x.bindFramebuffer(I.READ_FRAMEBUFFER,B),x.bindFramebuffer(I.DRAW_FRAMEBUFFER,z);for(let At=0;At<ye;At++)ba?I.framebufferTextureLayer(I.READ_FRAMEBUFFER,I.COLOR_ATTACHMENT0,Gt.__webglTexture,O,ke+At):I.framebufferTexture2D(I.READ_FRAMEBUFFER,I.COLOR_ATTACHMENT0,I.TEXTURE_2D,Gt.__webglTexture,O),mt?I.framebufferTextureLayer(I.DRAW_FRAMEBUFFER,I.COLOR_ATTACHMENT0,Ws.__webglTexture,pe,qt+At):I.framebufferTexture2D(I.DRAW_FRAMEBUFFER,I.COLOR_ATTACHMENT0,I.TEXTURE_2D,Ws.__webglTexture,pe),O!==0?I.blitFramebuffer(Ee,ze,xe,de,_e,ot,xe,de,I.COLOR_BUFFER_BIT,I.NEAREST):mt?I.copyTexSubImage3D(ve,pe,_e,ot,qt+At,Ee,ze,xe,de):I.copyTexSubImage2D(ve,pe,_e,ot,Ee,ze,xe,de);x.bindFramebuffer(I.READ_FRAMEBUFFER,null),x.bindFramebuffer(I.DRAW_FRAMEBUFFER,null)}else mt?A.isDataTexture||A.isData3DTexture?I.texSubImage3D(ve,pe,_e,ot,qt,xe,de,ye,vt,An,Ct.data):D.isCompressedArrayTexture?I.compressedTexSubImage3D(ve,pe,_e,ot,qt,xe,de,ye,vt,Ct.data):I.texSubImage3D(ve,pe,_e,ot,qt,xe,de,ye,vt,An,Ct):A.isDataTexture?I.texSubImage2D(I.TEXTURE_2D,pe,_e,ot,xe,de,vt,An,Ct.data):A.isCompressedTexture?I.compressedTexSubImage2D(I.TEXTURE_2D,pe,_e,ot,Ct.width,Ct.height,vt,Ct.data):I.texSubImage2D(I.TEXTURE_2D,pe,_e,ot,xe,de,vt,An,Ct);x.pixelStorei(I.UNPACK_ROW_LENGTH,In),x.pixelStorei(I.UNPACK_IMAGE_HEIGHT,et),x.pixelStorei(I.UNPACK_SKIP_PIXELS,di),x.pixelStorei(I.UNPACK_SKIP_ROWS,qi),x.pixelStorei(I.UNPACK_SKIP_IMAGES,ks),pe===0&&D.generateMipmaps&&I.generateMipmap(ve),x.unbindTexture()},this.initRenderTarget=function(A){H.get(A).__webglFramebuffer===void 0&&Y.setupRenderTarget(A)},this.initTexture=function(A){A.isCubeTexture?Y.setTextureCube(A,0):A.isData3DTexture?Y.setTexture3D(A,0):A.isDataArrayTexture||A.isCompressedArrayTexture?Y.setTexture2DArray(A,0):Y.setTexture2D(A,0),x.unbindTexture()},this.resetState=function(){Z=0,q=0,ie=null,x.reset(),me.reset()},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}get coordinateSystem(){return Gi}get outputColorSpace(){return this._outputColorSpace}set outputColorSpace(e){this._outputColorSpace=e;let n=this.getContext();n.drawingBufferColorSpace=Ke._getDrawingBufferColorSpace(e),n.unpackColorSpace=Ke._getUnpackColorSpace()}};var zU=(()=>{let t=new Float32Array([-1,-1,0,3,-1,0,-1,3,0]),e=new Float32Array([0,0,2,0,0,2]),n=new wi;return n.setAttribute("position",new Fn(t,3)),n.setAttribute("uv",new Fn(e,2)),n})(),zr=class tv{static get fullscreenGeometry(){return zU}constructor(e="Pass",n=new Dr,i=new Vs){this.name=e,this.renderer=null,this.scene=n,this.camera=i,this.screen=null,this.rtt=!0,this.needsSwap=!0,this.needsDepthBlit=!1,this.needsDepthTexture=!1,this.enabled=!0}get renderToScreen(){return!this.rtt}set renderToScreen(e){if(this.rtt===e){let n=this.fullscreenMaterial;n!==null&&(n.needsUpdate=!0),this.rtt=!e}}set mainScene(e){}set mainCamera(e){}setRenderer(e){this.renderer=e}isEnabled(){return this.enabled}setEnabled(e){this.enabled=e}get fullscreenMaterial(){return this.screen!==null?this.screen.material:null}set fullscreenMaterial(e){let n=this.screen;n!==null?n.material=e:(n=new Rn(tv.fullscreenGeometry,e),n.frustumCulled=!1,this.scene===null&&(this.scene=new Dr),this.scene.add(n),this.screen=n)}getFullscreenMaterial(){return this.fullscreenMaterial}setFullscreenMaterial(e){this.fullscreenMaterial=e}getDepthTexture(){return null}setDepthTexture(e,n=gs){}render(e,n,i,s,r){throw new Error("Render method not implemented!")}setSize(e,n){}initialize(e,n,i){}dispose(){for(let e of Object.keys(this)){let n=this[e];(n instanceof Ft||n instanceof Vi||n instanceof jt||n instanceof tv)&&this[e].dispose()}this.fullscreenMaterial!==null&&this.fullscreenMaterial.dispose()}},HU=class extends zr{constructor(){super("ClearMaskPass",null,null),this.needsSwap=!1}render(t,e,n,i,s){let r=t.state.buffers.stencil;r.setLocked(!1),r.setTest(!1)}},GU=`#ifdef COLOR_WRITE
#include <common>
#include <dithering_pars_fragment>
#ifdef FRAMEBUFFER_PRECISION_HIGH
uniform mediump sampler2D inputBuffer;
#else
uniform lowp sampler2D inputBuffer;
#endif
#endif
#ifdef DEPTH_WRITE
#include <packing>
#ifdef GL_FRAGMENT_PRECISION_HIGH
uniform highp sampler2D depthBuffer;
#else
uniform mediump sampler2D depthBuffer;
#endif
float readDepth(const in vec2 uv){
#if DEPTH_PACKING == 3201
return unpackRGBAToDepth(texture2D(depthBuffer,uv));
#else
return texture2D(depthBuffer,uv).r;
#endif
}
#endif
#ifdef USE_WEIGHTS
uniform vec4 channelWeights;
#endif
uniform float opacity;varying vec2 vUv;void main(){
#ifdef COLOR_WRITE
vec4 texel=texture2D(inputBuffer,vUv);
#ifdef USE_WEIGHTS
texel*=channelWeights;
#endif
gl_FragColor=opacity*texel;
#ifdef COLOR_SPACE_CONVERSION
#include <colorspace_fragment>
#endif
#include <dithering_fragment>
#else
gl_FragColor=vec4(0.0);
#endif
#ifdef DEPTH_WRITE
gl_FragDepth=readDepth(vUv);
#endif
}`,VU="varying vec2 vUv;void main(){vUv=position.xy*0.5+0.5;gl_Position=vec4(position.xy,1.0,1.0);}",kU=class extends Wt{constructor(){super({name:"CopyMaterial",defines:{COLOR_SPACE_CONVERSION:"1",DEPTH_PACKING:"0",COLOR_WRITE:"1"},uniforms:{inputBuffer:new Nt(null),depthBuffer:new Nt(null),channelWeights:new Nt(null),opacity:new Nt(1)},blending:Kn,toneMapped:!1,depthWrite:!1,depthTest:!1,fragmentShader:GU,vertexShader:VU}),this.depthFunc=Go}get inputBuffer(){return this.uniforms.inputBuffer.value}set inputBuffer(t){let e=t!==null;this.colorWrite!==e&&(e?this.defines.COLOR_WRITE=!0:delete this.defines.COLOR_WRITE,this.colorWrite=e,this.needsUpdate=!0),this.uniforms.inputBuffer.value=t}get depthBuffer(){return this.uniforms.depthBuffer.value}set depthBuffer(t){let e=t!==null;this.depthWrite!==e&&(e?this.defines.DEPTH_WRITE=!0:delete this.defines.DEPTH_WRITE,this.depthTest=e,this.depthWrite=e,this.needsUpdate=!0),this.uniforms.depthBuffer.value=t}set depthPacking(t){this.defines.DEPTH_PACKING=t.toFixed(0),this.needsUpdate=!0}get colorSpaceConversion(){return this.defines.COLOR_SPACE_CONVERSION!==void 0}set colorSpaceConversion(t){this.colorSpaceConversion!==t&&(t?this.defines.COLOR_SPACE_CONVERSION=!0:delete this.defines.COLOR_SPACE_CONVERSION,this.needsUpdate=!0)}get channelWeights(){return this.uniforms.channelWeights.value}set channelWeights(t){t!==null?(this.defines.USE_WEIGHTS="1",this.uniforms.channelWeights.value=t):delete this.defines.USE_WEIGHTS,this.needsUpdate=!0}setInputBuffer(t){this.uniforms.inputBuffer.value=t}getOpacity(t){return this.uniforms.opacity.value}setOpacity(t){this.uniforms.opacity.value=t}},WU=class extends zr{constructor(t,e=!0){super("CopyPass"),this.fullscreenMaterial=new kU,this.needsSwap=!1,this.renderTarget=t,t===void 0&&(this.renderTarget=new Ft(1,1,{minFilter:wt,magFilter:wt,stencilBuffer:!1,depthBuffer:!1}),this.renderTarget.texture.name="CopyPass.Target"),this.autoResize=e}get resize(){return this.autoResize}set resize(t){this.autoResize=t}get texture(){return this.renderTarget.texture}getTexture(){return this.renderTarget.texture}setAutoResizeEnabled(t){this.autoResize=t}render(t,e,n,i,s){this.fullscreenMaterial.inputBuffer=e.texture,t.setRenderTarget(this.renderToScreen?null:this.renderTarget),t.render(this.scene,this.camera)}setSize(t,e){this.autoResize&&this.renderTarget.setSize(t,e)}initialize(t,e,n){n!==void 0&&(this.renderTarget.texture.type=n,n!==Xt?this.fullscreenMaterial.defines.FRAMEBUFFER_PRECISION_HIGH="1":t!==null&&t.outputColorSpace===Dt&&(this.renderTarget.texture.colorSpace=Dt))}},vM=new Ye,_M=class extends zr{constructor(t=!0,e=!0,n=!1){super("ClearPass",null,null),this.needsSwap=!1,this.color=t,this.depth=e,this.stencil=n,this.overrideClearColor=null,this.overrideClearAlpha=-1}setClearFlags(t,e,n){this.color=t,this.depth=e,this.stencil=n}getOverrideClearColor(){return this.overrideClearColor}setOverrideClearColor(t){this.overrideClearColor=t}getOverrideClearAlpha(){return this.overrideClearAlpha}setOverrideClearAlpha(t){this.overrideClearAlpha=t}render(t,e,n,i,s){let r=this.overrideClearColor,a=this.overrideClearAlpha,o=t.getClearAlpha(),l=r!==null,c=a>=0;l?(t.getClearColor(vM),t.setClearColor(r,c?a:o)):c&&t.setClearAlpha(a),t.setRenderTarget(this.renderToScreen?null:e),t.clear(this.color,this.depth,this.stencil),l?t.setClearColor(vM,o):c&&t.setClearAlpha(o)}},XU=class extends zr{constructor(t,e){super("MaskPass",t,e),this.needsSwap=!1,this.clearPass=new _M(!1,!1,!0),this.inverse=!1}set mainScene(t){this.scene=t}set mainCamera(t){this.camera=t}get inverted(){return this.inverse}set inverted(t){this.inverse=t}get clear(){return this.clearPass.enabled}set clear(t){this.clearPass.enabled=t}getClearPass(){return this.clearPass}isInverted(){return this.inverted}setInverted(t){this.inverted=t}render(t,e,n,i,s){let r=t.getContext(),a=t.state.buffers,o=this.scene,l=this.camera,c=this.clearPass,h=this.inverted?0:1,p=1-h;a.color.setMask(!1),a.depth.setMask(!1),a.color.setLocked(!0),a.depth.setLocked(!0),a.stencil.setTest(!0),a.stencil.setOp(r.REPLACE,r.REPLACE,r.REPLACE),a.stencil.setFunc(r.ALWAYS,h,4294967295),a.stencil.setClear(p),a.stencil.setLocked(!0),this.clearPass.enabled&&(this.renderToScreen?c.render(t,null):(c.render(t,e),c.render(t,n))),this.renderToScreen?(t.setRenderTarget(null),t.render(o,l)):(t.setRenderTarget(e),t.render(o,l),t.setRenderTarget(n),t.render(o,l)),a.color.setLocked(!1),a.depth.setLocked(!1),a.stencil.setLocked(!1),a.stencil.setFunc(r.EQUAL,1,4294967295),a.stencil.setOp(r.KEEP,r.KEEP,r.KEEP),a.stencil.setLocked(!0)}};function YU(t,e){let n=t.getContext();if(e<=0||typeof n.renderbufferStorageMultisample!="function")return 0;let i=n.getParameter(n.MAX_SAMPLES),s=Math.min(e,i);if(s<=0)return 0;let r=n.getParameter(n.RENDERBUFFER_BINDING),a=n.createRenderbuffer();try{return n.bindRenderbuffer(n.RENDERBUFFER,a),n.renderbufferStorageMultisample(n.RENDERBUFFER,s,n.RGBA8,1,1),s}catch{return 0}finally{n.bindRenderbuffer(n.RENDERBUFFER,r),n.deleteRenderbuffer(a)}}var j0=1/1e3,qU=1e3,QU=class{constructor(){this.startTime=performance.now(),this.previousTime=0,this.currentTime=0,this._delta=0,this._elapsed=0,this._fixedDelta=1e3/60,this.timescale=1,this.useFixedDelta=!1,this._autoReset=!1}get autoReset(){return this._autoReset}set autoReset(t){typeof document<"u"&&document.hidden!==void 0&&(t?document.addEventListener("visibilitychange",this):document.removeEventListener("visibilitychange",this),this._autoReset=t)}get delta(){return this._delta*j0}get fixedDelta(){return this._fixedDelta*j0}set fixedDelta(t){this._fixedDelta=t*qU}get elapsed(){return this._elapsed*j0}update(t){this.useFixedDelta?this._delta=this.fixedDelta:(this.previousTime=this.currentTime,this.currentTime=(t!==void 0?t:performance.now())-this.startTime,this._delta=this.currentTime-this.previousTime),this._delta*=this.timescale,this._elapsed+=this._delta}reset(){this._delta=0,this._elapsed=0,this.currentTime=performance.now()-this.startTime}getDelta(){return this.delta}getElapsed(){return this.elapsed}handleEvent(t){document.hidden||(this.currentTime=performance.now()-this.startTime)}dispose(){this.autoReset=!1}},SM=class{constructor(t=null,{depthBuffer:e=!0,stencilBuffer:n=!1,multisampling:i=0,frameBufferType:s=Xt}={}){this.renderer=null,this.inputBuffer=this.createBuffer(e,n,s,i),this.outputBuffer=this.inputBuffer.clone(),this.copyPass=new WU,this.depthRenderTarget=null,this.passes=[],this.timer=new QU,this.autoRenderToScreen=!0,this.setRenderer(t)}get stableDepthTexture(){return this.depthRenderTarget===null?null:this.depthRenderTarget.depthTexture}get multisampling(){return this.inputBuffer.samples}set multisampling(t){let e=this.renderer===null?t:YU(this.renderer,t);this.multisampling!==e&&(this.inputBuffer.samples=e,this.outputBuffer.samples=e,this.inputBuffer.dispose(),this.outputBuffer.dispose())}getTimer(){return this.timer}getRenderer(){return this.renderer}setRenderer(t){if(this.renderer=t,t!==null){let e=t.getSize(new Ne),n=t.getContext().getContextAttributes().alpha,i=this.inputBuffer.texture.type;i===Xt&&t.outputColorSpace===Dt&&(this.inputBuffer.texture.colorSpace=Dt,this.outputBuffer.texture.colorSpace=Dt,this.inputBuffer.dispose(),this.outputBuffer.dispose());let s=this.multisampling;this.multisampling=s,t.autoClear=!1,this.setSize(e.width,e.height);for(let r of this.passes)r.initialize(t,n,i)}}replaceRenderer(t,e=!0){let n=this.renderer,i=n.domElement.parentNode;return this.setRenderer(t),e&&i!==null&&(i.removeChild(n.domElement),i.appendChild(t.domElement)),n}createDepthTexture(){let t=new Ci;t.name="EffectComposer.InputDepth",this.inputBuffer.stencilBuffer?(t.format=ms,t.type=Or):t.type=fi;let e=new Ci;e.format=t.format,e.type=t.type,e.name="EffectComposer.OutputDepth";let n=new Ci;n.format=t.format,n.type=t.type,n.name="EffectComposer.StableDepth",this.inputBuffer.depthTexture=t,this.outputBuffer.depthTexture=e,this.inputBuffer.dispose(),this.outputBuffer.dispose();let{width:i,height:s}=this.inputBuffer;this.depthRenderTarget=new Ft(i,s,{depthBuffer:!0,stencilBuffer:this.inputBuffer.stencilBuffer,depthTexture:n})}blitDepthBuffer(t){let e=this.renderer,n=this.depthRenderTarget,i=e.properties,s=e.getContext();e.setRenderTarget(n);let r=i.get(t).__webglFramebuffer,a=i.get(n).__webglFramebuffer,o=t.stencilBuffer?s.DEPTH_BUFFER_BIT|s.STENCIL_BUFFER_BIT:s.DEPTH_BUFFER_BIT;s.bindFramebuffer(s.READ_FRAMEBUFFER,r),s.bindFramebuffer(s.DRAW_FRAMEBUFFER,a),s.blitFramebuffer(0,0,t.width,t.height,0,0,n.width,n.height,o,s.NEAREST),s.bindFramebuffer(s.READ_FRAMEBUFFER,null),s.bindFramebuffer(s.DRAW_FRAMEBUFFER,null),e.setRenderTarget(null)}deleteDepthTexture(){let t=this.stableDepthTexture;for(let e of this.passes)e.getDepthTexture()===t&&e.setDepthTexture(null);this.depthRenderTarget!==null&&(this.depthRenderTarget.dispose(),this.depthRenderTarget=null),this.inputBuffer.depthTexture!==null&&(this.inputBuffer.depthTexture.dispose(),this.inputBuffer.depthTexture=null),this.outputBuffer.depthTexture!==null&&(this.outputBuffer.depthTexture.dispose(),this.outputBuffer.depthTexture=null)}createBuffer(t,e,n,i){let s=this.renderer,r=s===null?new Ne:s.getDrawingBufferSize(new Ne),a=new Ft(r.width,r.height,{minFilter:wt,magFilter:wt,samples:i,stencilBuffer:e,depthBuffer:t,type:n});return n===Xt&&s!==null&&s.outputColorSpace===Dt&&(a.texture.colorSpace=Dt),a.texture.name="EffectComposer.Buffer",a.texture.generateMipmaps=!1,a}setMainScene(t){for(let e of this.passes)e.mainScene=t}setMainCamera(t){for(let e of this.passes)e.mainCamera=t}addPass(t,e){let n=this.passes,i=this.renderer,s=i.getDrawingBufferSize(new Ne),r=i.getContext().getContextAttributes().alpha,a=this.inputBuffer.texture.type;if(t.renderer=i,t.setSize(s.width,s.height),t.initialize(i,r,a),this.autoRenderToScreen&&(n.length>0&&(n[n.length-1].renderToScreen=!1),t.renderToScreen&&(this.autoRenderToScreen=!1)),e!==void 0?n.splice(e,0,t):n.push(t),this.autoRenderToScreen&&(n[n.length-1].renderToScreen=!0),t.needsDepthTexture||this.depthRenderTarget!==null)if(this.depthRenderTarget===null){this.createDepthTexture();for(let o of n)o.setDepthTexture(this.stableDepthTexture)}else t.setDepthTexture(this.stableDepthTexture)}removePass(t){let e=this.passes,n=e.indexOf(t);if(n!==-1&&e.splice(n,1).length>0){let r=this.stableDepthTexture;if(r!==null){let a=(l,c)=>l||c.needsDepthTexture;e.reduce(a,!1)||(t.getDepthTexture()===r&&t.setDepthTexture(null),this.deleteDepthTexture())}this.autoRenderToScreen&&n===e.length&&(t.renderToScreen=!1,e.length>0&&(e[e.length-1].renderToScreen=!0))}}removeAllPasses(){let t=this.passes;this.deleteDepthTexture(),t.length>0&&(this.autoRenderToScreen&&(t[t.length-1].renderToScreen=!1),this.passes=[])}render(t){let e=this.renderer,n=this.copyPass,i=this.inputBuffer,s=this.outputBuffer,r,a=!1;t===void 0&&(this.timer.update(),t=this.timer.getDelta());for(let o of this.passes)if(o.enabled){if(o.render(e,i,s,t,a),o.needsDepthBlit&&this.depthRenderTarget!==null&&this.blitDepthBuffer(i),o.needsSwap){if(a){n.renderToScreen=o.renderToScreen;let l=e.getContext(),c=e.state.buffers.stencil;c.setFunc(l.NOTEQUAL,1,4294967295),n.render(e,i,s,t,a),c.setFunc(l.EQUAL,1,4294967295)}r=i,i=s,s=r}o instanceof XU?a=!0:o instanceof HU&&(a=!1)}}setSize(t,e,n){let i=this.renderer,s=i.getSize(new Ne);(t===void 0||e===void 0)&&(t=s.width,e=s.height),(s.width!==t||s.height!==e)&&i.setSize(t,e,n);let r=i.getDrawingBufferSize(new Ne);this.inputBuffer.setSize(r.width,r.height),this.outputBuffer.setSize(r.width,r.height),this.depthRenderTarget!==null&&this.depthRenderTarget.setSize(r.width,r.height);for(let a of this.passes)a.setSize(r.width,r.height)}reset(){this.dispose(),this.autoRenderToScreen=!0}dispose(){for(let t of this.passes)t.dispose();this.deleteDepthTexture(),this.inputBuffer.dispose(),this.outputBuffer.dispose(),this.copyPass.dispose(),this.timer.dispose(),this.passes=[],zr.fullscreenGeometry.dispose()}},Ta={NONE:0,DEPTH:1,CONVOLUTION:2},it={FRAGMENT_HEAD:"FRAGMENT_HEAD",FRAGMENT_MAIN_UV:"FRAGMENT_MAIN_UV",FRAGMENT_MAIN_IMAGE:"FRAGMENT_MAIN_IMAGE",VERTEX_HEAD:"VERTEX_HEAD",VERTEX_MAIN_SUPPORT:"VERTEX_MAIN_SUPPORT"},ZU=class{constructor(){this.shaderParts=new Map([[it.FRAGMENT_HEAD,null],[it.FRAGMENT_MAIN_UV,null],[it.FRAGMENT_MAIN_IMAGE,null],[it.VERTEX_HEAD,null],[it.VERTEX_MAIN_SUPPORT,null]]),this.defines=new Map,this.uniforms=new Map,this.blendModes=new Map,this.extensions=new Set,this.attributes=Ta.NONE,this.varyings=new Set,this.uvTransformation=!1,this.readDepth=!1,this.colorSpace=Gs}};var $0=!1,xM=class{constructor(t=null){this.originalMaterials=new Map,this.material=null,this.materials=null,this.materialsBackSide=null,this.materialsDoubleSide=null,this.materialsFlatShaded=null,this.materialsFlatShadedBackSide=null,this.materialsFlatShadedDoubleSide=null,this.setMaterial(t),this.meshCount=0,this.replaceMaterial=e=>{if(e.isMesh){let n;if(e.material.flatShading)switch(e.material.side){case zn:n=this.materialsFlatShadedDoubleSide;break;case $t:n=this.materialsFlatShadedBackSide;break;default:n=this.materialsFlatShaded;break}else switch(e.material.side){case zn:n=this.materialsDoubleSide;break;case $t:n=this.materialsBackSide;break;default:n=this.materials;break}this.originalMaterials.set(e,e.material),e.isSkinnedMesh?e.material=n[2]:e.isInstancedMesh?e.material=n[1]:e.material=n[0],++this.meshCount}}}cloneMaterial(t){if(!(t instanceof Wt))return t.clone();let e=t.uniforms,n=new Map;for(let s in e){let r=e[s].value;r.isRenderTargetTexture&&(e[s].value=null,n.set(s,r))}let i=t.clone();for(let s of n)e[s[0]].value=s[1],i.uniforms[s[0]].value=s[1];return i}setMaterial(t){if(this.disposeMaterials(),this.material=t,t!==null){let e=this.materials=[this.cloneMaterial(t),this.cloneMaterial(t),this.cloneMaterial(t)];for(let n of e)n.uniforms=Object.assign({},t.uniforms),n.side=ps;e[2].skinning=!0,this.materialsBackSide=e.map(n=>{let i=this.cloneMaterial(n);return i.uniforms=Object.assign({},t.uniforms),i.side=$t,i}),this.materialsDoubleSide=e.map(n=>{let i=this.cloneMaterial(n);return i.uniforms=Object.assign({},t.uniforms),i.side=zn,i}),this.materialsFlatShaded=e.map(n=>{let i=this.cloneMaterial(n);return i.uniforms=Object.assign({},t.uniforms),i.flatShading=!0,i}),this.materialsFlatShadedBackSide=e.map(n=>{let i=this.cloneMaterial(n);return i.uniforms=Object.assign({},t.uniforms),i.flatShading=!0,i.side=$t,i}),this.materialsFlatShadedDoubleSide=e.map(n=>{let i=this.cloneMaterial(n);return i.uniforms=Object.assign({},t.uniforms),i.flatShading=!0,i.side=zn,i})}}render(t,e,n){let i=t.shadowMap.enabled;if(t.shadowMap.enabled=!1,$0){let s=this.originalMaterials;this.meshCount=0,e.traverse(this.replaceMaterial),t.render(e,n);for(let r of s)r[0].material=r[1];this.meshCount!==s.size&&s.clear()}else{let s=e.overrideMaterial;e.overrideMaterial=this.material,t.render(e,n),e.overrideMaterial=s}t.shadowMap.enabled=i}disposeMaterials(){if(this.material!==null){let t=this.materials.concat(this.materialsBackSide).concat(this.materialsDoubleSide).concat(this.materialsFlatShaded).concat(this.materialsFlatShadedBackSide).concat(this.materialsFlatShadedDoubleSide);for(let e of t)e.dispose()}}dispose(){this.originalMaterials.clear(),this.disposeMaterials()}static get workaroundEnabled(){return $0}static set workaroundEnabled(t){$0=t}};var je={SKIP:9,SET:30,ADD:0,ALPHA:1,AVERAGE:2,COLOR:3,COLOR_BURN:4,COLOR_DODGE:5,DARKEN:6,DIFFERENCE:7,DIVIDE:8,DST:9,EXCLUSION:10,HARD_LIGHT:11,HARD_MIX:12,HUE:13,INVERT:14,INVERT_RGB:15,LIGHTEN:16,LINEAR_BURN:17,LINEAR_DODGE:18,LINEAR_LIGHT:19,LUMINOSITY:20,MULTIPLY:21,NEGATION:22,NORMAL:23,OVERLAY:24,PIN_LIGHT:25,REFLECT:26,SATURATION:27,SCREEN:28,SOFT_LIGHT:29,SRC:30,SUBTRACT:31,VIVID_LIGHT:32},KU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=dst.rgb+src.rgb;return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",JU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){return mix(dst,src,src.a*opacity);}",jU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=(dst.rgb+src.rgb)*0.5;return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",$U="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 a=RGBToHSL(dst.rgb);vec3 b=RGBToHSL(src.rgb);vec3 c=HSLToRGB(vec3(b.xy,a.z));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",e3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 a=dst.rgb,b=src.rgb;vec3 c=mix(step(0.0,b)*(1.0-min(vec3(1.0),(1.0-a)/max(b,1e-9))),vec3(1.0),step(1.0,a));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",t3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 a=dst.rgb,b=src.rgb;vec3 c=step(0.0,a)*mix(min(vec3(1.0),a/max(1.0-b,1e-9)),vec3(1.0),step(1.0,b));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",n3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=min(dst.rgb,src.rgb);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",i3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=abs(dst.rgb-src.rgb);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",s3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=dst.rgb/max(src.rgb,1e-9);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",r3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=dst.rgb+src.rgb-2.0*dst.rgb*src.rgb;return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",a3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 a=min(dst.rgb,1.0);vec3 b=min(src.rgb,1.0);vec3 c=mix(2.0*a*b,1.0-2.0*(1.0-a)*(1.0-b),step(0.5,b));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",o3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=step(1.0,dst.rgb+src.rgb);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",l3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 a=RGBToHSL(dst.rgb);vec3 b=RGBToHSL(src.rgb);vec3 c=HSLToRGB(vec3(b.x,a.yz));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",c3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=max(1.0-src.rgb,0.0);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",u3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=src.rgb*max(1.0-dst.rgb,0.0);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",f3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=max(dst.rgb,src.rgb);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",h3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=clamp(src.rgb+dst.rgb-1.0,0.0,1.0);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",d3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=min(dst.rgb+src.rgb,1.0);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",p3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=clamp(2.0*src.rgb+dst.rgb-1.0,0.0,1.0);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",m3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 a=RGBToHSL(dst.rgb);vec3 b=RGBToHSL(src.rgb);vec3 c=HSLToRGB(vec3(a.xy,b.z));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",g3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=dst.rgb*src.rgb;return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",v3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=max(1.0-abs(1.0-dst.rgb-src.rgb),0.0);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",x3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){return mix(dst,src,opacity);}",y3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 a=2.0*src.rgb*dst.rgb;vec3 b=1.0-2.0*(1.0-src.rgb)*(1.0-dst.rgb);vec3 c=mix(a,b,step(0.5,dst.rgb));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",_3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 src2=2.0*src.rgb;vec3 c=mix(mix(src2,dst.rgb,step(0.5*dst.rgb,src.rgb)),max(src2-1.0,vec3(0.0)),step(dst.rgb,src2-1.0));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",S3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 a=min(dst.rgb*dst.rgb/max(1.0-src.rgb,1e-9),1.0);vec3 c=mix(a,src.rgb,step(1.0,src.rgb));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",A3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 a=RGBToHSL(dst.rgb);vec3 b=RGBToHSL(src.rgb);vec3 c=HSLToRGB(vec3(a.x,b.y,a.z));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",M3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=dst.rgb+src.rgb-min(dst.rgb*src.rgb,1.0);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",E3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 src2=2.0*src.rgb;vec3 d=dst.rgb+(src2-1.0);vec3 w=step(0.5,src.rgb);vec3 a=dst.rgb-(1.0-src2)*dst.rgb*(1.0-dst.rgb);vec3 b=mix(d*(sqrt(dst.rgb)-dst.rgb),d*dst.rgb*((16.0*dst.rgb-12.0)*dst.rgb+3.0),w*(1.0-step(0.25,dst.rgb)));vec3 c=mix(a,b,w);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",T3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){return src;}",b3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=max(dst.rgb-src.rgb,0.0);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",w3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=mix(max(1.0-min((1.0-dst.rgb)/(2.0*src.rgb),1.0),0.0),min(dst.rgb/(2.0*(1.0-src.rgb)),1.0),step(0.5,src.rgb));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",C3=new Map([[je.ADD,KU],[je.ALPHA,JU],[je.AVERAGE,jU],[je.COLOR,$U],[je.COLOR_BURN,e3],[je.COLOR_DODGE,t3],[je.DARKEN,n3],[je.DIFFERENCE,i3],[je.DIVIDE,s3],[je.DST,null],[je.EXCLUSION,r3],[je.HARD_LIGHT,a3],[je.HARD_MIX,o3],[je.HUE,l3],[je.INVERT,c3],[je.INVERT_RGB,u3],[je.LIGHTEN,f3],[je.LINEAR_BURN,h3],[je.LINEAR_DODGE,d3],[je.LINEAR_LIGHT,p3],[je.LUMINOSITY,m3],[je.MULTIPLY,g3],[je.NEGATION,v3],[je.NORMAL,x3],[je.OVERLAY,y3],[je.PIN_LIGHT,_3],[je.REFLECT,S3],[je.SATURATION,A3],[je.SCREEN,M3],[je.SOFT_LIGHT,E3],[je.SRC,T3],[je.SUBTRACT,b3],[je.VIVID_LIGHT,w3]]),R3=class extends Zn{constructor(t,e=1){super(),this._blendFunction=t,this.opacity=new Nt(e)}getOpacity(){return this.opacity.value}setOpacity(t){this.opacity.value=t}get blendFunction(){return this._blendFunction}set blendFunction(t){this._blendFunction=t,this.dispatchEvent({type:"change"})}getBlendFunction(){return this.blendFunction}setBlendFunction(t){this.blendFunction=t}getShaderCode(){return C3.get(this.blendFunction)}};var AM=class extends Zn{constructor(t,e,{attributes:n=Ta.NONE,blendFunction:i=je.NORMAL,defines:s=new Map,uniforms:r=new Map,extensions:a=null,vertexShader:o=null}={}){super(),this.name=t,this.renderer=null,this.attributes=n,this.fragmentShader=e,this.vertexShader=o,this.defines=s,this.uniforms=r,this.extensions=a,this.blendMode=new R3(i),this.blendMode.addEventListener("change",l=>this.setChanged()),this._inputColorSpace=Gs,this._outputColorSpace=hi}get inputColorSpace(){return this._inputColorSpace}set inputColorSpace(t){this._inputColorSpace=t,this.setChanged()}get outputColorSpace(){return this._outputColorSpace}set outputColorSpace(t){this._outputColorSpace=t,this.setChanged()}set mainScene(t){}set mainCamera(t){}getName(){return this.name}setRenderer(t){this.renderer=t}getDefines(){return this.defines}getUniforms(){return this.uniforms}getExtensions(){return this.extensions}getBlendMode(){return this.blendMode}getAttributes(){return this.attributes}setAttributes(t){this.attributes=t,this.setChanged()}getFragmentShader(){return this.fragmentShader}setFragmentShader(t){this.fragmentShader=t,this.setChanged()}getVertexShader(){return this.vertexShader}setVertexShader(t){this.vertexShader=t,this.setChanged()}setChanged(){this.dispatchEvent({type:"change"})}setDepthTexture(t,e=gs){}update(t,e,n){}setSize(t,e){}initialize(t,e,n){}dispose(){for(let t of Object.keys(this)){let e=this[t];(e instanceof Ft||e instanceof Vi||e instanceof jt||e instanceof zr)&&this[t].dispose()}}};var ZP=[new Float32Array([0,0]),new Float32Array([0,1,1]),new Float32Array([0,1,1,2]),new Float32Array([0,1,2,2,3]),new Float32Array([0,1,2,3,4,4,5]),new Float32Array([0,1,2,3,4,5,7,8,9,10])];var MM=class extends zr{constructor(t,e,n=null){super("RenderPass",t,e),this.needsSwap=!1,this.needsDepthBlit=!0,this.clearPass=new _M,this.overrideMaterialManager=n===null?null:new xM(n),this.ignoreBackground=!1,this.skipShadowMapUpdate=!1,this.selection=null}set mainScene(t){this.scene=t}set mainCamera(t){this.camera=t}get renderToScreen(){return super.renderToScreen}set renderToScreen(t){super.renderToScreen=t,this.clearPass.renderToScreen=t}get overrideMaterial(){let t=this.overrideMaterialManager;return t!==null?t.material:null}set overrideMaterial(t){let e=this.overrideMaterialManager;t!==null?e!==null?e.setMaterial(t):this.overrideMaterialManager=new xM(t):e!==null&&(e.dispose(),this.overrideMaterialManager=null)}getOverrideMaterial(){return this.overrideMaterial}setOverrideMaterial(t){this.overrideMaterial=t}get clear(){return this.clearPass.enabled}set clear(t){this.clearPass.enabled=t}getSelection(){return this.selection}setSelection(t){this.selection=t}isBackgroundDisabled(){return this.ignoreBackground}setBackgroundDisabled(t){this.ignoreBackground=t}isShadowMapDisabled(){return this.skipShadowMapUpdate}setShadowMapDisabled(t){this.skipShadowMapUpdate=t}getClearPass(){return this.clearPass}render(t,e,n,i,s){let r=this.scene,a=this.camera,o=this.selection,l=a.layers.mask,c=r.background,h=t.shadowMap.autoUpdate,p=this.renderToScreen?null:e;o!==null&&a.layers.set(o.getLayer()),this.skipShadowMapUpdate&&(t.shadowMap.autoUpdate=!1),(this.ignoreBackground||this.clearPass.overrideClearColor!==null)&&(r.background=null),this.clearPass.enabled&&this.clearPass.render(t,e),t.setRenderTarget(p),this.overrideMaterialManager!==null?this.overrideMaterialManager.render(t,r,a):t.render(r,a),a.layers.mask=l,r.background=c,t.shadowMap.autoUpdate=h}};var KP=Math.PI*.5;var D3=`#include <common>
#include <packing>
#include <dithering_pars_fragment>
#define packFloatToRGBA(v) packDepthToRGBA(v)
#define unpackRGBAToFloat(v) unpackRGBAToDepth(v)
#ifdef FRAMEBUFFER_PRECISION_HIGH
uniform mediump sampler2D inputBuffer;
#else
uniform lowp sampler2D inputBuffer;
#endif
#if DEPTH_PACKING == 3201
uniform lowp sampler2D depthBuffer;
#elif defined(GL_FRAGMENT_PRECISION_HIGH)
uniform highp sampler2D depthBuffer;
#else
uniform mediump sampler2D depthBuffer;
#endif
uniform vec2 resolution;uniform vec2 texelSize;uniform float cameraNear;uniform float cameraFar;uniform float aspect;uniform float time;varying vec2 vUv;vec4 sRGBToLinear(const in vec4 value){return vec4(mix(pow(value.rgb*0.9478672986+vec3(0.0521327014),vec3(2.4)),value.rgb*0.0773993808,vec3(lessThanEqual(value.rgb,vec3(0.04045)))),value.a);}float readDepth(const in vec2 uv){
#if DEPTH_PACKING == 3201
float depth=unpackRGBAToDepth(texture2D(depthBuffer,uv));
#else
float depth=texture2D(depthBuffer,uv).r;
#endif
#if defined(USE_LOGARITHMIC_DEPTH_BUFFER) || defined(LOG_DEPTH)
float d=pow(2.0,depth*log2(cameraFar+1.0))-1.0;float a=cameraFar/(cameraFar-cameraNear);float b=cameraFar*cameraNear/(cameraNear-cameraFar);depth=a+b/d;
#elif defined(USE_REVERSED_DEPTH_BUFFER)
depth=1.0-depth;
#endif
return depth;}float getViewZ(const in float depth){
#ifdef PERSPECTIVE_CAMERA
return perspectiveDepthToViewZ(depth,cameraNear,cameraFar);
#else
return orthographicDepthToViewZ(depth,cameraNear,cameraFar);
#endif
}vec3 RGBToHCV(const in vec3 RGB){vec4 P=mix(vec4(RGB.bg,-1.0,2.0/3.0),vec4(RGB.gb,0.0,-1.0/3.0),step(RGB.b,RGB.g));vec4 Q=mix(vec4(P.xyw,RGB.r),vec4(RGB.r,P.yzx),step(P.x,RGB.r));float C=Q.x-min(Q.w,Q.y);float H=abs((Q.w-Q.y)/(6.0*C+EPSILON)+Q.z);return vec3(H,C,Q.x);}vec3 RGBToHSL(const in vec3 RGB){vec3 HCV=RGBToHCV(RGB);float L=HCV.z-HCV.y*0.5;float S=HCV.y/(1.0-abs(L*2.0-1.0)+EPSILON);return vec3(HCV.x,S,L);}vec3 HueToRGB(const in float H){float R=abs(H*6.0-3.0)-1.0;float G=2.0-abs(H*6.0-2.0);float B=2.0-abs(H*6.0-4.0);return clamp(vec3(R,G,B),0.0,1.0);}vec3 HSLToRGB(const in vec3 HSL){vec3 RGB=HueToRGB(HSL.x);float C=(1.0-abs(2.0*HSL.z-1.0))*HSL.y;return(RGB-0.5)*C+HSL.z;}FRAGMENT_HEAD void main(){FRAGMENT_MAIN_UV vec4 color0=texture2D(inputBuffer,UV);vec4 color1=vec4(0.0);FRAGMENT_MAIN_IMAGE color0.a=clamp(color0.a,0.0,1.0);gl_FragColor=color0;
#ifdef ENCODE_OUTPUT
#include <colorspace_fragment>
#endif
#include <dithering_fragment>
}`,U3="uniform vec2 resolution;uniform vec2 texelSize;uniform float cameraNear;uniform float cameraFar;uniform float aspect;uniform float time;varying vec2 vUv;VERTEX_HEAD void main(){vUv=position.xy*0.5+0.5;VERTEX_MAIN_SUPPORT gl_Position=vec4(position.xy,1.0,1.0);}",I3=class extends Wt{constructor(t,e,n,i,s=!1){super({name:"EffectMaterial",defines:{THREE_REVISION:"186".replace(/\D+/g,""),DEPTH_PACKING:"0",ENCODE_OUTPUT:"1"},uniforms:{inputBuffer:new Nt(null),depthBuffer:new Nt(null),resolution:new Nt(new Ne),texelSize:new Nt(new Ne),cameraNear:new Nt(.3),cameraFar:new Nt(1e3),aspect:new Nt(1),time:new Nt(0)},blending:Kn,toneMapped:!1,depthWrite:!1,depthTest:!1,dithering:s}),t&&this.setShaderParts(t),e&&this.setDefines(e),n&&this.setUniforms(n),this.copyCameraSettings(i)}set inputBuffer(t){this.uniforms.inputBuffer.value=t}setInputBuffer(t){this.uniforms.inputBuffer.value=t}get depthBuffer(){return this.uniforms.depthBuffer.value}set depthBuffer(t){this.uniforms.depthBuffer.value=t}get depthPacking(){return Number(this.defines.DEPTH_PACKING)}set depthPacking(t){this.defines.DEPTH_PACKING=t.toFixed(0),this.needsUpdate=!0}setDepthBuffer(t,e=gs){this.depthBuffer=t,this.depthPacking=e}setShaderData(t){this.setShaderParts(t.shaderParts),this.setDefines(t.defines),this.setUniforms(t.uniforms),this.setExtensions(t.extensions)}setShaderParts(t){return this.fragmentShader=D3.replace(it.FRAGMENT_HEAD,t.get(it.FRAGMENT_HEAD)||"").replace(it.FRAGMENT_MAIN_UV,t.get(it.FRAGMENT_MAIN_UV)||"").replace(it.FRAGMENT_MAIN_IMAGE,t.get(it.FRAGMENT_MAIN_IMAGE)||""),this.vertexShader=U3.replace(it.VERTEX_HEAD,t.get(it.VERTEX_HEAD)||"").replace(it.VERTEX_MAIN_SUPPORT,t.get(it.VERTEX_MAIN_SUPPORT)||""),this.needsUpdate=!0,this}setDefines(t){for(let e of t.entries())this.defines[e[0]]=e[1];return this.needsUpdate=!0,this}setUniforms(t){for(let e of t.entries())this.uniforms[e[0]]=e[1];return this}setExtensions(t){this.extensions={};for(let e of t)this.extensions[e]=!0;return this}get encodeOutput(){return this.defines.ENCODE_OUTPUT!==void 0}set encodeOutput(t){this.encodeOutput!==t&&(t?this.defines.ENCODE_OUTPUT="1":delete this.defines.ENCODE_OUTPUT,this.needsUpdate=!0)}isOutputEncodingEnabled(t){return this.encodeOutput}setOutputEncodingEnabled(t){this.encodeOutput=t}get time(){return this.uniforms.time.value}set time(t){this.uniforms.time.value=t}setDeltaTime(t){this.uniforms.time.value+=t}adoptCameraSettings(t){this.copyCameraSettings(t)}copyCameraSettings(t){t&&(this.uniforms.cameraNear.value=t.near,this.uniforms.cameraFar.value=t.far,t instanceof Cn?this.defines.PERSPECTIVE_CAMERA="1":delete this.defines.PERSPECTIVE_CAMERA,this.needsUpdate=!0)}setSize(t,e){let n=this.uniforms;n.resolution.value.set(t,e),n.texelSize.value.set(1/t,1/e),n.aspect.value=t/e}static get Section(){return it}};var jP=Number("186".replace(/\D+/g,"")),Ea=255/256,$P=new Float32Array([Ea/256**3,Ea/256**2,Ea/256,Ea]),eL=new Float32Array([Ea,Ea/256,Ea/256**2,1/256**3]);function yM(t,e,n){for(let i of e){let s="$1"+t+i.charAt(0).toUpperCase()+i.slice(1),r=new RegExp("([^\\.])(\\b"+i+"\\b)","g");for(let a of n.entries())a[1]!==null&&n.set(a[0],a[1].replace(r,s))}}function B3(t,e,n){let i=e.getFragmentShader(),s=e.getVertexShader(),r=i!==void 0&&/mainImage/.test(i),a=i!==void 0&&/mainUv/.test(i);if(n.attributes|=e.getAttributes(),i===void 0)throw new Error(`Missing fragment shader (${e.name})`);if(a&&(n.attributes&Ta.CONVOLUTION)!==0)throw new Error(`Effects that transform UVs are incompatible with convolution effects (${e.name})`);if(!r&&!a)throw new Error(`Could not find mainImage or mainUv function (${e.name})`);{let o=/\w+\s+(\w+)\([\w\s,]*\)\s*{/g,l=n.shaderParts,c=l.get(it.FRAGMENT_HEAD)||"",h=l.get(it.FRAGMENT_MAIN_UV)||"",p=l.get(it.FRAGMENT_MAIN_IMAGE)||"",u=l.get(it.VERTEX_HEAD)||"",d=l.get(it.VERTEX_MAIN_SUPPORT)||"",v=new Set,M=new Set;if(a&&(h+=`	${t}MainUv(UV);
`,n.uvTransformation=!0),s!==null&&/mainSupport/.test(s)){let g=/mainSupport *\([\w\s]*?uv\s*?\)/.test(s);d+=`	${t}MainSupport(`,d+=g?`vUv);
`:`);
`;for(let S of s.matchAll(/(?:varying\s+\w+\s+([\S\s]*?);)/g))for(let _ of S[1].split(/\s*,\s*/))n.varyings.add(_),v.add(_),M.add(_);for(let S of s.matchAll(o))M.add(S[1])}for(let g of i.matchAll(o))M.add(g[1]);for(let g of e.defines.keys())M.add(g.replace(/\([\w\s,]*\)/g,""));for(let g of e.uniforms.keys())M.add(g);M.delete("while"),M.delete("for"),M.delete("if"),e.uniforms.forEach((g,S)=>n.uniforms.set(t+S.charAt(0).toUpperCase()+S.slice(1),g)),e.defines.forEach((g,S)=>n.defines.set(t+S.charAt(0).toUpperCase()+S.slice(1),g));let m=new Map([["fragment",i],["vertex",s]]);yM(t,M,n.defines),yM(t,M,m),i=m.get("fragment"),s=m.get("vertex");let f=e.blendMode;if(n.blendModes.set(f.blendFunction,f),r){e.inputColorSpace!==null&&e.inputColorSpace!==n.colorSpace&&(p+=e.inputColorSpace===Dt?`color0 = sRGBTransferOETF(color0);
	`:`color0 = sRGBToLinear(color0);
	`),e.outputColorSpace!==hi?n.colorSpace=e.outputColorSpace:e.inputColorSpace!==null&&(n.colorSpace=e.inputColorSpace);let g=/MainImage *\([\w\s,]*?depth[\w\s,]*?\)/;p+=`${t}MainImage(color0, UV, `,(n.attributes&Ta.DEPTH)!==0&&g.test(i)&&(p+="depth, ",n.readDepth=!0),p+=`color1);
	`;let S=t+"BlendOpacity";n.uniforms.set(S,f.opacity),p+=`color0 = blend${f.blendFunction}(color0, color1, ${S});

	`,c+=`uniform float ${S};

`}if(c+=i+`
`,s!==null&&(u+=s+`
`),l.set(it.FRAGMENT_HEAD,c),l.set(it.FRAGMENT_MAIN_UV,h),l.set(it.FRAGMENT_MAIN_IMAGE,p),l.set(it.VERTEX_HEAD,u),l.set(it.VERTEX_MAIN_SUPPORT,d),e.extensions!==null)for(let g of e.extensions)n.extensions.add(g)}}var EM=class extends zr{constructor(t,...e){super("EffectPass"),this.fullscreenMaterial=new I3(null,null,null,t),this.listener=n=>this.handleEvent(n),this.effects=[],this.setEffects(e),this.skipRendering=!1,this.minTime=1,this.maxTime=Number.POSITIVE_INFINITY,this.timeScale=1}set mainScene(t){for(let e of this.effects)e.mainScene=t}set mainCamera(t){this.fullscreenMaterial.copyCameraSettings(t);for(let e of this.effects)e.mainCamera=t}get encodeOutput(){return this.fullscreenMaterial.encodeOutput}set encodeOutput(t){this.fullscreenMaterial.encodeOutput=t}get dithering(){return this.fullscreenMaterial.dithering}set dithering(t){let e=this.fullscreenMaterial;e.dithering=t,e.needsUpdate=!0}setEffects(t){for(let e of this.effects)e.removeEventListener("change",this.listener);this.effects=t.sort((e,n)=>n.attributes-e.attributes);for(let e of this.effects)e.addEventListener("change",this.listener)}updateMaterial(){let t=new ZU,e=0;for(let a of this.effects)if(a.blendMode.blendFunction===je.DST)t.attributes|=a.getAttributes()&Ta.DEPTH;else{if((t.attributes&a.getAttributes()&Ta.CONVOLUTION)!==0)throw new Error(`Convolution effects cannot be merged (${a.name})`);B3("e"+e++,a,t)}let n=t.shaderParts.get(it.FRAGMENT_HEAD),i=t.shaderParts.get(it.FRAGMENT_MAIN_IMAGE),s=t.shaderParts.get(it.FRAGMENT_MAIN_UV),r=/\bblend\b/g;for(let a of t.blendModes.values())n+=a.getShaderCode().replace(r,`blend${a.blendFunction}`)+`
`;(t.attributes&Ta.DEPTH)!==0?(t.readDepth&&(i=`float depth = readDepth(UV);

	`+i),this.needsDepthTexture=this.getDepthTexture()===null):this.needsDepthTexture=!1,t.colorSpace===Dt&&(i+=`color0 = sRGBToLinear(color0);
	`),t.uvTransformation?(s=`vec2 transformedUv = vUv;
`+s,t.defines.set("UV","transformedUv")):t.defines.set("UV","vUv"),t.shaderParts.set(it.FRAGMENT_HEAD,n),t.shaderParts.set(it.FRAGMENT_MAIN_IMAGE,i),t.shaderParts.set(it.FRAGMENT_MAIN_UV,s);for(let[a,o]of t.shaderParts)o!==null&&t.shaderParts.set(a,o.trim().replace(/^#/,`
#`));this.skipRendering=e===0,this.needsSwap=!this.skipRendering,this.fullscreenMaterial.setShaderData(t)}recompile(){this.updateMaterial()}getDepthTexture(){return this.fullscreenMaterial.depthBuffer}setDepthTexture(t,e=gs){this.fullscreenMaterial.depthBuffer=t,this.fullscreenMaterial.depthPacking=e;for(let n of this.effects)n.setDepthTexture(t,e)}render(t,e,n,i,s){for(let r of this.effects)r.update(t,e,i);if(!this.skipRendering||this.renderToScreen){let r=this.fullscreenMaterial;r.inputBuffer=e.texture,r.time+=i*this.timeScale,t.setRenderTarget(this.renderToScreen?null:n),t.render(this.scene,this.camera)}}setSize(t,e){this.fullscreenMaterial.setSize(t,e);for(let n of this.effects)n.setSize(t,e)}initialize(t,e,n){this.renderer=t;for(let i of this.effects)i.initialize(t,e,n);this.updateMaterial(),n!==void 0&&n!==Xt&&(this.fullscreenMaterial.defines.FRAMEBUFFER_PRECISION_HIGH="1")}dispose(){super.dispose();for(let t of this.effects)t.removeEventListener("change",this.listener),t.dispose()}handleEvent(t){t.type==="change"&&this.recompile()}};var nL=[new Float32Array(3),new Float32Array(3)],iL=[new Float32Array(3),new Float32Array(3),new Float32Array(3),new Float32Array(3)],sL=[[new Float32Array([0,0,0]),new Float32Array([1,0,0]),new Float32Array([1,1,0]),new Float32Array([1,1,1])],[new Float32Array([0,0,0]),new Float32Array([1,0,0]),new Float32Array([1,0,1]),new Float32Array([1,1,1])],[new Float32Array([0,0,0]),new Float32Array([0,0,1]),new Float32Array([1,0,1]),new Float32Array([1,1,1])],[new Float32Array([0,0,0]),new Float32Array([0,1,0]),new Float32Array([1,1,0]),new Float32Array([1,1,1])],[new Float32Array([0,0,0]),new Float32Array([0,1,0]),new Float32Array([0,1,1]),new Float32Array([1,1,1])],[new Float32Array([0,0,0]),new Float32Array([0,0,1]),new Float32Array([0,1,1]),new Float32Array([1,1,1])]];var rL=[new Float32Array(2),new Float32Array(2)];var aL=new Float32Array([0,-.25,.25,-.125,.125,-.375,.375]),oL=[new Float32Array([0,0]),new Float32Array([.25,-.25]),new Float32Array([-.25,.25]),new Float32Array([.125,-.125]),new Float32Array([-.125,.125])],lL=[new Uint8Array([0,0]),new Uint8Array([3,0]),new Uint8Array([0,3]),new Uint8Array([3,3]),new Uint8Array([1,0]),new Uint8Array([4,0]),new Uint8Array([1,3]),new Uint8Array([4,3]),new Uint8Array([0,1]),new Uint8Array([3,1]),new Uint8Array([0,4]),new Uint8Array([3,4]),new Uint8Array([1,1]),new Uint8Array([4,1]),new Uint8Array([1,4]),new Uint8Array([4,4])],cL=[new Uint8Array([0,0]),new Uint8Array([1,0]),new Uint8Array([0,2]),new Uint8Array([1,2]),new Uint8Array([2,0]),new Uint8Array([3,0]),new Uint8Array([2,2]),new Uint8Array([3,2]),new Uint8Array([0,1]),new Uint8Array([1,1]),new Uint8Array([0,3]),new Uint8Array([1,3]),new Uint8Array([2,1]),new Uint8Array([3,1]),new Uint8Array([2,3]),new Uint8Array([3,3])];var uL=new Map([[Un(0,0,0,0),new Float32Array([0,0,0,0])],[Un(0,0,0,1),new Float32Array([0,0,0,1])],[Un(0,0,1,0),new Float32Array([0,0,1,0])],[Un(0,0,1,1),new Float32Array([0,0,1,1])],[Un(0,1,0,0),new Float32Array([0,1,0,0])],[Un(0,1,0,1),new Float32Array([0,1,0,1])],[Un(0,1,1,0),new Float32Array([0,1,1,0])],[Un(0,1,1,1),new Float32Array([0,1,1,1])],[Un(1,0,0,0),new Float32Array([1,0,0,0])],[Un(1,0,0,1),new Float32Array([1,0,0,1])],[Un(1,0,1,0),new Float32Array([1,0,1,0])],[Un(1,0,1,1),new Float32Array([1,0,1,1])],[Un(1,1,0,0),new Float32Array([1,1,0,0])],[Un(1,1,0,1),new Float32Array([1,1,0,1])],[Un(1,1,1,0),new Float32Array([1,1,1,0])],[Un(1,1,1,1),new Float32Array([1,1,1,1])]]);function ev(t,e,n){return t+(e-t)*n}function Un(t,e,n,i){let s=ev(t,e,.75),r=ev(n,i,1-.25);return ev(s,r,1-.125)}var Md=Gr(tl());var CM=Gr(Fc()),L3=()=>{let e=document.createElement("canvas");e.width=64,e.height=64;let n=e.getContext("2d");if(!n)throw new Error("2D context not available");n.fillStyle="black",n.fillRect(0,0,e.width,e.height);let i=new jt(e);i.minFilter=wt,i.magFilter=wt,i.generateMipmaps=!1;let s=[],r=null,a=64,o=.1*64,l=1/a,c=()=>{n.fillStyle="black",n.fillRect(0,0,e.width,e.height)},h=d=>{let v={x:d.x*64,y:(1-d.y)*64},M=1,m=_=>Math.sin(_*Math.PI/2),f=_=>-_*(_-2);d.age<a*.3?M=m(d.age/(a*.3)):M=f(1-(d.age-a*.3)/(a*.7))||0,M*=d.force;let g=`${(d.vx+1)/2*255}, ${(d.vy+1)/2*255}, ${M*255}`,S=320;n.shadowOffsetX=S,n.shadowOffsetY=S,n.shadowBlur=o,n.shadowColor=`rgba(${g},${.22*M})`,n.beginPath(),n.fillStyle="rgba(255,0,0,1)",n.arc(v.x-S,v.y-S,o,0,Math.PI*2),n.fill()};return{texture:i,addTouch:d=>{let v=0,M=0,m=0;if(r){let f=d.x-r.x,g=d.y-r.y;if(f===0&&g===0)return;let S=f*f+g*g,_=Math.sqrt(S);M=f/(_||1),m=g/(_||1),v=Math.min(S*1e4,1)}r={x:d.x,y:d.y},s.push({x:d.x,y:d.y,age:0,force:v,vx:M,vy:m})},update:()=>{c();for(let d=s.length-1;d>=0;d-=1){let v=s[d],M=v.force*l*(1-v.age/a);v.x+=v.vx*M,v.y+=v.vy*M,v.age+=1,v.age>a&&s.splice(d,1)}s.forEach(h),i.needsUpdate=!0},set radiusScale(d){o=.1*64*d}}},O3=(t,e)=>{let n=`
    uniform sampler2D uTexture;
    uniform float uStrength;
    uniform float uTime;
    uniform float uFreq;

    void mainUv(inout vec2 uv) {
      vec4 tex = texture2D(uTexture, uv);
      float vx = tex.r * 2.0 - 1.0;
      float vy = tex.g * 2.0 - 1.0;
      float intensity = tex.b;
      float wave = 0.5 + 0.5 * sin(uTime * uFreq + intensity * 6.2831853);
      float amount = uStrength * intensity * wave;
      uv += vec2(vx, vy) * amount;
    }
  `;return new AM("LiquidEffect",n,{uniforms:new Map([["uTexture",new Nt(t)],["uStrength",new Nt(e?.strength??.025)],["uTime",new Nt(0)],["uFreq",new Nt(e?.freq??4.5)]])})},F3={square:0,circle:1,triangle:2,diamond:3},z3=`
void main() {
  gl_Position = vec4(position, 1.0);
}
`,H3=`
precision highp float;

uniform vec3  uColor;
uniform vec2  uResolution;
uniform float uTime;
uniform float uPixelSize;
uniform float uScale;
uniform float uDensity;
uniform float uPixelJitter;
uniform int   uEnableRipples;
uniform float uRippleSpeed;
uniform float uRippleThickness;
uniform float uRippleIntensity;
uniform float uEdgeFade;
uniform int   uShapeType;

const int SHAPE_SQUARE   = 0;
const int SHAPE_CIRCLE   = 1;
const int SHAPE_TRIANGLE = 2;
const int SHAPE_DIAMOND  = 3;
const int MAX_CLICKS = 10;

uniform vec2  uClickPos[MAX_CLICKS];
uniform float uClickTimes[MAX_CLICKS];

out vec4 fragColor;

float Bayer2(vec2 a) {
  a = floor(a);
  return fract(a.x / 2.0 + a.y * a.y * 0.75);
}
#define Bayer4(a) (Bayer2(0.5*(a))*0.25 + Bayer2(a))
#define Bayer8(a) (Bayer4(0.5*(a))*0.25 + Bayer2(a))

#define FBM_OCTAVES 5
#define FBM_LACUNARITY 1.25
#define FBM_GAIN 1.0

float hash11(float n) {
  return fract(sin(n) * 43758.5453);
}

float vnoise(vec3 p) {
  vec3 ip = floor(p);
  vec3 fp = fract(p);
  float n000 = hash11(dot(ip + vec3(0.0,0.0,0.0), vec3(1.0,57.0,113.0)));
  float n100 = hash11(dot(ip + vec3(1.0,0.0,0.0), vec3(1.0,57.0,113.0)));
  float n010 = hash11(dot(ip + vec3(0.0,1.0,0.0), vec3(1.0,57.0,113.0)));
  float n110 = hash11(dot(ip + vec3(1.0,1.0,0.0), vec3(1.0,57.0,113.0)));
  float n001 = hash11(dot(ip + vec3(0.0,0.0,1.0), vec3(1.0,57.0,113.0)));
  float n101 = hash11(dot(ip + vec3(1.0,0.0,1.0), vec3(1.0,57.0,113.0)));
  float n011 = hash11(dot(ip + vec3(0.0,1.0,1.0), vec3(1.0,57.0,113.0)));
  float n111 = hash11(dot(ip + vec3(1.0,1.0,1.0), vec3(1.0,57.0,113.0)));
  vec3 w = fp*fp*fp*(fp*(fp*6.0-15.0)+10.0);
  float x00 = mix(n000, n100, w.x);
  float x10 = mix(n010, n110, w.x);
  float x01 = mix(n001, n101, w.x);
  float x11 = mix(n011, n111, w.x);
  float y0 = mix(x00, x10, w.y);
  float y1 = mix(x01, x11, w.y);
  return mix(y0, y1, w.z) * 2.0 - 1.0;
}

float fbm2(vec2 uv, float time) {
  vec3 p = vec3(uv * uScale, time);
  float amplitude = 1.0;
  float frequency = 1.0;
  float sum = 1.0;
  for (int index = 0; index < FBM_OCTAVES; ++index) {
    sum += amplitude * vnoise(p * frequency);
    frequency *= FBM_LACUNARITY;
    amplitude *= FBM_GAIN;
  }
  return sum * 0.5 + 0.5;
}

float maskCircle(vec2 p, float coverage) {
  float radius = sqrt(coverage) * 0.25;
  float distance = length(p - 0.5) - radius;
  float aa = 0.5 * fwidth(distance);
  return coverage * (1.0 - smoothstep(-aa, aa, distance * 2.0));
}

float maskTriangle(vec2 p, vec2 id, float coverage) {
  bool flip = mod(id.x + id.y, 2.0) > 0.5;
  if (flip) p.x = 1.0 - p.x;
  float radius = sqrt(coverage);
  float distance = p.y - radius * (1.0 - p.x);
  float aa = fwidth(distance);
  return coverage * clamp(0.5 - distance / aa, 0.0, 1.0);
}

float maskDiamond(vec2 p, float coverage) {
  float radius = sqrt(coverage) * 0.564;
  return step(abs(p.x - 0.49) + abs(p.y - 0.49), radius);
}

void main() {
  float pixelSize = uPixelSize;
  vec2 fragCoord = gl_FragCoord.xy - uResolution * 0.5;
  float aspectRatio = uResolution.x / uResolution.y;
  vec2 pixelId = floor(fragCoord / pixelSize);
  vec2 pixelUV = fract(fragCoord / pixelSize);
  float cellPixelSize = 8.0 * pixelSize;
  vec2 cellId = floor(fragCoord / cellPixelSize);
  vec2 cellCoord = cellId * cellPixelSize;
  vec2 uv = cellCoord / uResolution * vec2(aspectRatio, 1.0);

  float base = fbm2(uv, uTime * 0.05);
  base = base * 0.5 - 0.65;
  float feed = base + (uDensity - 0.5) * 0.3;

  if (uEnableRipples == 1) {
    for (int index = 0; index < MAX_CLICKS; ++index) {
      vec2 position = uClickPos[index];
      if (position.x < 0.0) continue;
      vec2 clickUv = (
        (position - uResolution * 0.5 - cellPixelSize * 0.5) / uResolution
      ) * vec2(aspectRatio, 1.0);
      float elapsed = max(uTime - uClickTimes[index], 0.0);
      float radius = distance(uv, clickUv);
      float waveRadius = uRippleSpeed * elapsed;
      float ring = exp(-pow((radius - waveRadius) / uRippleThickness, 2.0));
      float attenuation = exp(-elapsed) * exp(-10.0 * radius);
      feed = max(feed, ring * attenuation * uRippleIntensity);
    }
  }

  float bayer = Bayer8(fragCoord / uPixelSize) - 0.5;
  float blackWhite = step(0.5, feed + bayer);
  float hash = fract(
    sin(dot(floor(fragCoord / uPixelSize), vec2(127.1, 311.7))) * 43758.5453
  );
  float jitterScale = 1.0 + (hash - 0.5) * uPixelJitter;
  float coverage = blackWhite * jitterScale;
  float mask;
  if (uShapeType == SHAPE_CIRCLE) {
    mask = maskCircle(pixelUV, coverage);
  } else if (uShapeType == SHAPE_TRIANGLE) {
    mask = maskTriangle(pixelUV, pixelId, coverage);
  } else if (uShapeType == SHAPE_DIAMOND) {
    mask = maskDiamond(pixelUV, coverage);
  } else {
    mask = coverage;
  }

  if (uEdgeFade > 0.0) {
    vec2 normalized = gl_FragCoord.xy / uResolution;
    float edge = min(
      min(normalized.x, normalized.y),
      min(1.0 - normalized.x, 1.0 - normalized.y)
    );
    mask *= smoothstep(0.0, uEdgeFade, edge);
  }

  vec3 srgbColor = mix(
    uColor * 12.92,
    1.055 * pow(uColor, vec3(1.0 / 2.4)) - 0.055,
    step(0.0031308, uColor)
  );
  fragColor = vec4(srgbColor, mask);
}
`,nv=10;function iv({variant:t="square",pixelSize:e=4,color:n="#B497CF",className:i="",style:s,antialias:r=!0,patternScale:a=2,patternDensity:o=1,liquid:l=!1,liquidStrength:c=.1,liquidRadius:h=1,pixelSizeJitter:p=0,enableRipples:u=!0,rippleIntensityScale:d=1,rippleThickness:v=.1,rippleSpeed:M=.3,liquidWobbleSpeed:m=4.5,autoPauseOffscreen:f=!0,speed:g=.5,transparent:S=!0,edgeFade:_=.5}){let E=(0,Md.useRef)(null);return(0,Md.useEffect)(()=>{let T=E.current;if(!T)return;let C=document.createElement("canvas"),y=new xd({canvas:C,antialias:r,alpha:!0,powerPreference:"high-performance"});y.domElement.style.width="100%",y.domElement.style.height="100%",y.setPixelRatio(Math.min(window.devicePixelRatio||1,2)),T.appendChild(y.domElement),S?y.setClearAlpha(0):y.setClearColor(0,1);let b={uResolution:{value:new Ne(0,0)},uTime:{value:0},uColor:{value:new Ye(n)},uClickPos:{value:Array.from({length:nv},()=>new Ne(-1,-1))},uClickTimes:{value:new Float32Array(nv)},uShapeType:{value:F3[t]??0},uPixelSize:{value:e*y.getPixelRatio()},uScale:{value:a},uDensity:{value:o},uPixelJitter:{value:p},uEnableRipples:{value:u?1:0},uRippleSpeed:{value:M},uRippleThickness:{value:v},uRippleIntensity:{value:d},uEdgeFade:{value:_}},R=new Dr,N=new Vs(-1,1,1,-1,0,1),F=new Wt({vertexShader:z3,fragmentShader:H3,uniforms:b,transparent:!0,depthTest:!1,depthWrite:!1,glslVersion:Bc}),k=new Rn(new ya(2,2),F);R.add(k);let B,z,Z;if(l){z=L3(),z.radiusScale=h,B=new SM(y),B.addPass(new MM(R,N)),Z=O3(z.texture,{strength:c,freq:m});let ue=new EM(N,Z);ue.renderToScreen=!0,B.addPass(ue)}let q=()=>{let ue=T.clientWidth||1,Se=T.clientHeight||1;y.setSize(ue,Se,!1),b.uResolution.value.set(y.domElement.width,y.domElement.height),b.uPixelSize.value=e*y.getPixelRatio(),B?.setSize(y.domElement.width,y.domElement.height)},ie=new ResizeObserver(q);ie.observe(T),q();let W=ue=>{let Se=y.domElement.getBoundingClientRect(),ge=y.domElement.width/Se.width,Fe=y.domElement.height/Se.height;return{x:(ue.clientX-Se.left)*ge,y:(Se.height-(ue.clientY-Se.top))*Fe,width:y.domElement.width,height:y.domElement.height}},$=0,te=ue=>{let Se=W(ue);b.uClickPos.value[$].set(Se.x,Se.y),b.uClickTimes.value[$]=b.uTime.value,$=($+1)%nv},we=ue=>{if(!z)return;let Se=W(ue);z.addTouch({x:Se.x/Se.width,y:Se.y/Se.height})};window.addEventListener("pointerdown",te,{passive:!0}),window.addEventListener("pointermove",we,{passive:!0});let Me=!0,ut=()=>{Me=!document.hidden};document.addEventListener("visibilitychange",ut);let qe=new Mc,$e=Math.random()*1e3,X=0,ee=()=>{X=requestAnimationFrame(ee),!(f&&!Me)&&(b.uTime.value=$e+qe.getElapsedTime()*g,Z&&(Z.uniforms.get("uTime").value=b.uTime.value),B?(z?.update(),B.render()):y.render(R,N))};return X=requestAnimationFrame(ee),()=>{cancelAnimationFrame(X),ie.disconnect(),window.removeEventListener("pointerdown",te),window.removeEventListener("pointermove",we),document.removeEventListener("visibilitychange",ut),k.geometry.dispose(),F.dispose(),B?.dispose(),y.dispose(),y.forceContextLoss(),y.domElement.parentElement===T&&T.removeChild(y.domElement)}},[r,f,n,_,u,l,h,c,m,o,a,e,p,d,M,v,g,S,t]),(0,CM.jsx)("div",{ref:E,className:`pixel-blast-container ${i}`,style:s,"aria-hidden":"true"})}var sv=Gr(Fc());function G3(){let[t,e]=(0,Ed.useState)(document.documentElement.dataset.theme==="light");return(0,Ed.useEffect)(()=>{let n=new MutationObserver(()=>{e(document.documentElement.dataset.theme==="light")});return n.observe(document.documentElement,{attributes:!0,attributeFilter:["data-theme"]}),()=>n.disconnect()},[]),(0,sv.jsx)(iv,{variant:"square",pixelSize:4,color:t?"#000000":"#ffffff",patternScale:2,patternDensity:1.4,pixelSizeJitter:.5,enableRipples:!0,rippleSpeed:.4,rippleThickness:.12,rippleIntensityScale:1.5,liquid:!1,speed:.5,edgeFade:0,transparent:!0})}var RM=document.getElementById("pixel-blast-root");RM&&(0,DM.createRoot)(RM).render((0,sv.jsx)(G3,{}));})();
/*! Bundled license information:

react/cjs/react.production.js:
  (**
   * @license React
   * react.production.js
   *
   * Copyright (c) Meta Platforms, Inc. and affiliates.
   *
   * This source code is licensed under the MIT license found in the
   * LICENSE file in the root directory of this source tree.
   *)

scheduler/cjs/scheduler.production.js:
  (**
   * @license React
   * scheduler.production.js
   *
   * Copyright (c) Meta Platforms, Inc. and affiliates.
   *
   * This source code is licensed under the MIT license found in the
   * LICENSE file in the root directory of this source tree.
   *)

react-dom/cjs/react-dom.production.js:
  (**
   * @license React
   * react-dom.production.js
   *
   * Copyright (c) Meta Platforms, Inc. and affiliates.
   *
   * This source code is licensed under the MIT license found in the
   * LICENSE file in the root directory of this source tree.
   *)

react-dom/cjs/react-dom-client.production.js:
  (**
   * @license React
   * react-dom-client.production.js
   *
   * Copyright (c) Meta Platforms, Inc. and affiliates.
   *
   * This source code is licensed under the MIT license found in the
   * LICENSE file in the root directory of this source tree.
   *)

react/cjs/react-jsx-runtime.production.js:
  (**
   * @license React
   * react-jsx-runtime.production.js
   *
   * Copyright (c) Meta Platforms, Inc. and affiliates.
   *
   * This source code is licensed under the MIT license found in the
   * LICENSE file in the root directory of this source tree.
   *)

three/build/three.core.js:
three/build/three.module.js:
  (**
   * @license
   * Copyright 2010-2026 Three.js Authors
   * SPDX-License-Identifier: MIT
   *)

postprocessing/build/index.js:
  (**
   * postprocessing v6.39.5 build Wed Sep 09 2026
   * https://github.com/pmndrs/postprocessing
   * Copyright 2015-2026 Raoul van Rüschen
   * @license Zlib
   *)
*/
